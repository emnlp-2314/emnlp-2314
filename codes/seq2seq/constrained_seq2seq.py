#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
from multiprocessing import shared_memory
import random
import json
import logging
import os
from numpy.core.records import array
from sqlalchemy import false
import sklearn
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Callable, Dict, Tuple, Any, Optional
import numpy as np
from torch.cuda.amp import autocast
import math
from transformers import (
    PreTrainedTokenizer,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import (
    speed_metrics,
)
from transformers.trainer_pt_utils import SequentialDistributedSampler
from extraction.event_schema import EventSchema
from extraction.extract_constraint import get_constraint_decoder
from extraction.extraction_metrics import get_extract_metrics
from seq2seq.label_smoother_sum import SumLabelSmoother
from seq2seq.utils import lmap
import numpy as np
import collections
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import pickle
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter
# Integrations must be imported before ML frameworks:

import numpy as np
import torch
from packaging import version
from torch import cosine_similarity, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository
from transformers.file_utils import (
    is_datasets_available,)
    
if is_datasets_available():
    import datasets

from transformers import logging
logger = logging.get_logger(__name__)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,)

def add_logging_file(training_args):
    fh = logging.FileHandler(os.path.join(training_args.output_dir.rstrip(os.sep) + '.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def decode_tree_str(sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
                    tokenizer: PreTrainedTokenizer) -> List[str]:
    def clean_tree_text(x):
        return x.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()

    sequences = np.where(sequences != -100, sequences, tokenizer.pad_token_id)

    str_list = tokenizer.batch_decode(sequences, skip_special_tokens=False)
    return lmap(clean_tree_text, str_list)


def build_compute_extract_metrics_event_fn(decoding_type_schema: EventSchema,
                                           decoding_format: str,
                                           tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        return decode_tree_str(pred.predictions, tokenizer), decode_tree_str(pred.label_ids, tokenizer)

    def extraction_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        extraction = get_extract_metrics(pred_lns=pred_str, tgt_lns=label_str, label_constraint=decoding_type_schema,
                                         decoding_format=decoding_format)
        # rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        extraction.update({"gen_len": summ_len})
        # extraction.update( )
        return extraction

    compute_metrics_fn = extraction_metrics
    return compute_metrics_fn


@dataclass
class ConstraintSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Parameters:
        constraint_decoding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use Constraint Decoding
        structure_weight (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    loss_dir: str = field(default="./", metadata={"help": "Whether to Constraint Decoding or not."})
    NER: int = field(default=0, metadata={"help": "Whether to Constraint Decoding or not."})
    RE: int = field(default=0, metadata={"help": "Whether to Constraint Decoding or not."})
    both: int = field(default=0, metadata={"help": "Whether to Constraint Decoding or not."})
    fine_grit_share_parm: int = field(default=0, metadata={"help": "Whether to Constraint Decoding or not."})
    cut_reverse: int = field(default=0, metadata={"help": "Whether to Constraint Decoding or not."})
    share_num: int = field(default=12, metadata={"help": "Whether to Constraint Decoding or not."})
    file_name: str = field(default='empty', metadata={"help": "File name to save gradients"})
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    label_smoothing_sum: bool = field(default=False,
                                      metadata={"help": "Whether to use sum token loss for label smoothing"})


class ConstraintSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, multi_task_metrics=None, decoding_type_schema=None, decoding_format='tree',source_prefix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_task_metrics = multi_task_metrics
        self.decoding_format = decoding_format
        self.decoding_type_schema = decoding_type_schema
        self.grad_dic = {'0':{}, '1':{}, '2':{}}
        self.grad_dic_all = {'0':{}, '1':{}, '2':{}}
        self.grad_drop_dic = {}
        self.step_num = 0
        self.loss_step = 0
        self.epoch = 0
        self.sim_dic = {}
        self.good_samples = {}
        self.bad_samples = {}
        self.calculate_gradient = 0
        self.calculate_gradient_fine_grit = 0
        self.fine_grit_share_parm = self.args.fine_grit_share_parm
        self.NER = self.args.NER
        self.RE = self.args.RE
        self.both = self.args.both
        self.grad_drop = 0
        self.save_sample = 0
        self.multi_decoder = 0
        self.share_num = self.args.share_num
        self.cut_reverse = self.args.cut_reverse
        self.writer = SummaryWriter(log_dir= self.args.loss_dir + "/summary_pic")
        self.sim_file = './seq2seq/' + kwargs['args'].file_name + '_grad_sim.json'
        self.good_file = './seq2seq/' + kwargs['args'].file_name + '_good.json'
        self.bad_file = './seq2seq/' + kwargs['args'].file_name + '_bad.json'
        self.grad_file = './seq2seq/' + kwargs['args'].file_name + '_all_grad.pickle'
        # Label smoothing by sum token loss, different from different Label smootheing
        if self.args.label_smoothing_sum and self.args.label_smoothing_factor != 0:
            self.label_smoother = SumLabelSmoother(epsilon=self.args.label_smoothing_factor)
            print('Using %s' % self.label_smoother)
        elif self.args.label_smoothing_factor != 0:
            print('Using %s' % self.label_smoother)
        else:
            self.label_smoother = None

        if self.args.constraint_decoding:
            self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                             type_schema=self.decoding_type_schema,
                                                             decoding_schema=self.decoding_format,
                                                             source_prefix=source_prefix)
        else:
            self.constraint_decoder = None
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if self.constraint_decoder else None,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # print(self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False))
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels

    @staticmethod
    def random_mask(inputs):
        """
        random mask input_ids
        """
        for i in range(len(inputs['input_ids'])):
            array1 = np.random.rand(len(inputs['input_ids'][i]))
            for idx in range(len(inputs['input_ids'][i])):
                if array1[idx] >= 0.95 and inputs['input_ids'][i][idx] not in inputs['labels'][i]:
                    inputs['input_ids'][i][idx] = 32000
        
        return inputs


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            """
            Perform a training step on a batch of inputs.

            Subclass and override to inject custom behavior.

            Args:
                model (:obj:`nn.Module`):
                    The model to train.
                inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument :obj:`labels`. Check your model's documentation for all accepted arguments.

            Return:
                :obj:`torch.Tensor`: The tensor with training loss on this batch.
            """
            if self.calculate_gradient:
                if self.step_num == 0: # 初始化梯度的和
                    for name, parms in model.named_parameters():
                        if 'encoder' in name:
                            self.grad_dic['0'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                            self.grad_dic['1'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                            self.grad_dic['2'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)

            if self.calculate_gradient_fine_grit:
                if self.step_num == 0: # 初始化梯度的和
                    for name, parms in model.named_parameters():
                        self.grad_dic['0'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                        self.grad_dic['1'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)

            if self.grad_drop or self.save_sample: # 使用滑动平均来记录主任务的梯度
                if self.step_num == 0: # 初始化梯度的和
                    for name, parms in model.named_parameters():
                        if 'encoder' in name:
                            self.grad_drop_dic[name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
            
            model.train()
            inputs = self._prepare_inputs(inputs)
            device = model.device
            # inputs = inputs.to(device)
            # inputs = self.random_mask(inputs)
            if self.calculate_gradient or self.grad_drop or self.save_sample: 
                # 如果计算梯度或者drop梯度或者记录梯度样本，需要记录上一步的梯度
                if self.step_num == 0:
                    last_step_grad = {}
                    for name, parms in model.named_parameters():
                        if 'encoder' in name:
                            last_step_grad[name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                else:
                    last_step_grad = {}
                    for name, parms in model.named_parameters():
                        if 'encoder' in name:
                            last_step_grad[name] = parms.grad.clone().detach()
            if self.calculate_gradient_fine_grit:
                if self.step_num == 0:
                    last_step_grad = {}
                    for name, parms in model.named_parameters():
                        last_step_grad[name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                else:
                    last_step_grad = {}
                    for name, parms in model.named_parameters():
                        if parms.grad == None:
                            continue
                        last_step_grad[name] = parms.grad.data.clone().detach()

            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                if self.multi_decoder:
                    task_map = {'<extra_id_32>': 0, '<extra_id_31>': 1, '<extra_id_33>': 1}
                    task_index = []
                    for ids in inputs['input_ids']:
                        tokens = self.tokenizer.convert_ids_to_tokens(list(ids))
                        for token in tokens:
                            for value, key in task_map.items():
                                if token == value:
                                    task_index.append(key)
                    if task_index[0] != 0: # '<extra_id_32>': 0 是主任务
                        if task_index[0] == 1:
                            # inputs['is_NER'] = True
                            inputs['is_RE'] = True
                        if task_index[0] == 2:
                            inputs['is_RE'] = True

                if self.fine_grit_share_parm:
                    task_map = {'<extra_id_32>': 0, '<extra_id_31>': 1, '<extra_id_33>': 1}
                    task_index = []
                    for ids in inputs['input_ids']:
                        tokens = self.tokenizer.convert_ids_to_tokens(list(ids))
                        for token in tokens:
                            for value, key in task_map.items():
                                if token == value:
                                    task_index.append(key)
                    # print(task_index)
                    if task_index[0] != 0: # '<extra_id_32>': 0 是主任务
                        if task_index[0] == 1:
                            inputs['is_RE'] = True
                        if task_index[0] == 2:
                            inputs['is_RE'] = True
                        inputs['is_fine_grid_share_parm'] = True

                loss = self.compute_loss(model, inputs)


            task_map = {'<extra_id_32>': 0, '<extra_id_31>': 1, '<extra_id_33>': 1}
            task_index = []
            for ids in inputs['input_ids']:
                tokens = self.tokenizer.convert_ids_to_tokens(list(ids))
                for token in tokens:
                    for value, key in task_map.items():
                        if token == value:
                            task_index.append(key)


            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                pass
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

            # if self.fine_grit_share_parm:
            #     if task_index[0] != 0: # '<extra_id_32>': 0 是主任务
            #         if task_index[0] == 1:
            #             inputs['is_RE'] = True
            #         if task_index[0] == 2:
            #             inputs['is_RE'] = True
            #         inputs['is_fine_grid_share_parm'] = True
            #         print('AUX')
            #         for name, parms in model.named_parameters():
            #             if 'decoder.block.9.layer.2.DenseReluDense.wo.weight' in name:
            #                 if parms.grad == None:
            #                     print(name, 'None')
            #                 else:
            #                     print(name, parms.grad.data[-1][-1])
            #     else:
            #         print('TARGET')
            #         for name, parms in model.named_parameters():
            #             if 'decoder.block.9.layer.2.DenseReluDense.wo.weight' in name:
            #                 if parms.grad == None:
            #                     print(name, 'None')
            #                 else:
            #                     print(name, parms.grad.data[-1][-1])

            
            def is_no_zero(arr):
                for i in range(0, len(arr)):
                    if arr[i] != 0:
                        return True
                return False
            
            if self.save_sample:
                task_map = {'<extra_id_32>': 0, '<extra_id_31>': 1, '<extra_id_33>': 1}
                task_index = []
                for ids in inputs['input_ids']:
                    tokens = self.tokenizer.convert_ids_to_tokens(list(ids))
                    for token in tokens:
                        for value, key in task_map.items():
                            if token == value:
                                task_index.append(key)
                current_step_grad = {}
                for name, parms in model.named_parameters():
                    if 'encoder' in name:
                        current_step_grad[name] = parms.grad.clone().detach()
                if (self.step_num / self.args.gradient_accumulation_steps) % self.args.eval_steps == 0: # 一个新的epoch
                    self.epoch += 1
                    self.good_samples[self.epoch] = []
                    self.bad_samples[self.epoch] = []
                if task_index[0] != 0: # 辅助任务
                    tmp1 = 0
                    tmp2 = 0
                    tmp3 = 0
                    for key, value1 in self.grad_drop_dic.items():
                        value2 = current_step_grad[key] - last_step_grad[key]
                        tmp1 += torch.sum((value1 * value2).reshape(1, -1))
                        tmp2 += torch.sum((value1 * value1).reshape(1, -1))
                        tmp3 += torch.sum((value2 * value2).reshape(1, -1))
                    sim = tmp1 / (math.sqrt(tmp2) * math.sqrt(tmp3) + 1e-8)
                    sample = random.random()
                    # 梯度相似度较高或者较低都记录样本

                    num = self.step_num % ( self.args.gradient_accumulation_steps * self.args.eval_steps )
                    if sim > 0.2:
                        self.good_samples[self.epoch].append(num)
                    if sim < -0.2:
                        self.bad_samples[self.epoch].append(num)
                else: # 主任务，使用EMA
                    beta = 1e-2
                    for name, parms in model.named_parameters():
                        if 'encoder' in name:
                            # print("value1",self.grad_drop_dic[name])
                            self.grad_drop_dic[name] = (1 - beta) * self.grad_drop_dic[name] + beta * (current_step_grad[name] - last_step_grad[name])
            self.step_num += 1
            if (task_index[0] == 0):
                self.writer.add_scalar("loss", loss.detach(), self.loss_step)
                self.loss_step += 1
            if (self.step_num == (self.args.eval_steps * 4 * 50)):
                self.writer.close()
            if self.fine_grit_share_parm and self.step_num % 4 == 0:
                # RE
                if self.RE:
                    grad_list = ['encoder.block.1', 'decoder.block.1', 'encoder.block.9', 'decoder.block.2', 'decoder.block.9', 'encoder.block.11', 'encoder.block.8', 'decoder.block.6', 'encoder.block.4', 'encoder.block.8', 'encoder.block.10', 'decoder.block.7', 'decoder.block.10', 'encoder.block.6', 'encoder.block.5', 'encoder.block.3', 'encoder.block.7', 'decoder.block.5', 'encoder.block.2', 'decoder.block.3', 'encoder.block.0', 'decoder.block.0', 'decoder.block.4', 'decoder.block.11']
                if self.NER:
                    grad_list = ['encoder.block.1', 'decoder.block.1', 'encoder.block.4', 'decoder.block.2', 'encoder.block.7', 'encoder.block.10', 'encoder.block.8', 'encoder.block.6', 'encoder.block.5', 'encoder.block.9', 'decoder.block.0', 'encoder.block.3', 'encoder.block.11', 'encoder.block.2', 'decoder.block.10', 'decoder.block.8', 'decoder.block.7', 'decoder.block.9', 'encoder.block.0', 'decoder.block.6', 'decoder.block.3', 'decoder.block.4', 'decoder.block.11', 'decoder.block.5']
                # coreNLP NER
                # grad_list = ['encoder.block.1', 'decoder.block.1', 'decoder.block.2', 'encoder.block.9', 'encoder.block.4', 'encoder.block.8', 'encoder.block.7', 'encoder.block.3', 'encoder.block.10', 'encoder.block.6', 'encoder.block.5', 'decoder.block.0', 'encoder.block.2', 'encoder.block.11', 'decoder.block.8', 'decoder.block.10', 'encoder.block.0', 'decoder.block.3', 'decoder.block.9', 'decoder.block.6', 'decoder.block.7', 'decoder.block.4', 'decoder.block.11', 'decoder.block.5']
                # coreNLP RE
                # grad_list = ['encoder.block.1', 'encoder.block.3', 'encoder.block.10', 'encoder.block.11', 'encoder.block.2', 'encoder.block.4', 'encoder.block.5', 'encoder.block.6', 'encoder.block.8', 'encoder.block.0', 'decoder.block.8', 'encoder.block.7', 'decoder.block.9', 'encoder.block.9', 'decoder.block.3', 'decoder.block.2', 'decoder.block.7', 'decoder.block.6', 'decoder.block.10', 'decoder.block.0', 'decoder.block.5', 'decoder.block.4', 'decoder.block.1', 'decoder.block.11']
                if self.both:
                    # random
                    # grad_list = ['decoder.block.2', 'decoder.block.4', 'encoder.block.9', 'decoder.block.8', 'encoder.block.0', 'encoder.block.3', 'decoder.block.3', 'decoder.block.1', 'decoder.block.5', 'decoder.block.0', 'encoder.block.5', 'decoder.block.9', 'encoder.block.11', 'encoder.block.2', 'decoder.block.7', 'decoder.block.6', 'encoder.block.7', 'encoder.block.4', 'decoder.block.11', 'encoder.block.6', 'encoder.block.1', 'encoder.block.8', 'decoder.block.10', 'encoder.block.10']
                    # True
                    # grad_list = ['encoder.block.1', 'decoder.block.1', 'decoder.block.2', 'encoder.block.9', 'encoder.block.8', 'encoder.block.4', 'encoder.block.7', 'encoder.block.3', 'encoder.block.5', 'encoder.block.10', 'encoder.block.6', 'encoder.block.11', 'decoder.block.0', 'encoder.block.2', 'encoder.block.0', 'decoder.block.8', 'decoder.block.10', 'decoder.block.9', 'decoder.block.7', 'decoder.block.6', 'decoder.block.4', 'decoder.block.3', 'decoder.block.5', 'decoder.block.11']
                    grad_list = ['encoder.block.1', 'decoder.block.1', 'encoder.block.4', 'encoder.block.9', 'encoder.block.3', 'encoder.block.7', 'encoder.block.8', 'encoder.block.5', 'encoder.block.10', 'encoder.block.6', 'decoder.block.2', 'encoder.block.2', 'encoder.block.11', 'encoder.block.0', 'decoder.block.8', 'decoder.block.0', 'decoder.block.10', 'decoder.block.9', 'decoder.block.7', 'decoder.block.6', 'decoder.block.4', 'decoder.block.3', 'decoder.block.5', 'decoder.block.11']
                not_share_num = 24 - self.share_num
                if self.cut_reverse == 1:
                    not_shared_parms = grad_list[len(grad_list) - not_share_num: ]
                else:
                    not_shared_parms = grad_list[0: not_share_num]
                tmp_dic = {}
                for name1, parms1 in model.named_parameters():
                    tmp_dic[name1] = parms1
                for name, parms in model.named_parameters():

                    if name == 'shared.weight':
                        continue
                    if task_index[0] == 1: # NER
                        # if 'decoder.block.9.layer.2.DenseReluDense.wo.weight' in name:
                        #     print(name, parms.grad[-1][-1])
                        if 'NER_' not in name and 'RE_' not in name:
                            share_flag = 1
                            for i in not_shared_parms:
                                if i in name:
                                    share_flag = 0
                            if share_flag == 1:
                                if parms.grad == None:
                                    continue
                                
                                parms.grad.data += tmp_dic['RE_' + name].grad.data.clone().detach()
                                tmp_dic['RE_' + name].grad.data = parms.grad.data.clone().detach()
                                # if 'decoder.block.9.layer.2.DenseReluDense.wo.weight' in name:
                                #     print('main---', parms.grad[-1][-1])

                    if task_index[0] == 0: #DEE
                        # if 'decoder.block.9.layer.2.DenseReluDense.wo.weight' in name:
                        #     print(name, parms.grad[-1][-1])
                        if 'RE_' in name:
                            share_flag = 1
                            for i in not_shared_parms:
                                if i in name:
                                    share_flag = 0
                            if share_flag == 1:
                                if parms.grad == None:
                                    continue

                                parms.grad.data += tmp_dic[name[3:]].grad.data.clone().detach()
                                tmp_dic[name[3:]].grad.data = parms.grad.data.clone().detach()
                #                 if 'decoder.block.9.layer.2.DenseReluDense.wo.weight' in name:
                #                     print('aux---', parms.grad[-1][-1])
                # for name, parms in model.named_parameters():
                #     if 'decoder.block.9.layer.2.DenseReluDense.wo.weight' in name:
                #             print('???', name, parms.grad[-1][-1])
            if self.save_sample and self.epoch == self.args.num_train_epochs and (
                self.step_num / self.args.gradient_accumulation_steps) % self.args.eval_steps == 0: 
                # 到达最后一个epoch，保存每个epoch的梯度相似度和最后的总梯度
                good_file = open(self.good_file, "w")
                bad_file = open(self.bad_file, "w")
                json.dump(self.good_samples, good_file)
                json.dump(self.bad_samples, bad_file)
                good_file.close()
                bad_file.close()

            if self.calculate_gradient:
                for name, parms in model.named_parameters():
                    if 'encoder' in name:
                        self.grad_dic[str(task_index[0])][name] += (parms.grad.clone().detach() - last_step_grad[name])
                    # self.grad_dic_all[str(task_index[0])][name] += parms.grad.clone().detach()
                if (self.step_num / self.args.gradient_accumulation_steps) % self.args.eval_steps == 0: # 过了一个eval_step
                    self.epoch += 1
                    self.sim_dic[str(self.epoch)] = {}
                    for i in range(0, 2): # 计算梯度的余弦相似度
                        for j in range(i + 1, 3):
                            # print(task[i], task[j])
                            tmp1 = 0
                            tmp2 = 0
                            tmp3 = 0
                            for key, value1 in self.grad_dic[str(i)].items():
                                value2 = self.grad_dic[str(j)][key]
                                tmp1 += torch.sum((value1 * value2).reshape(1, -1))
                                tmp2 += torch.sum((value1 * value1).reshape(1, -1))
                                tmp3 += torch.sum((value2 * value2).reshape(1, -1))
                            sim = tmp1 / (math.sqrt(tmp2) * math.sqrt(tmp3) + 1e-8)
                            self.sim_dic[str(self.epoch)][str(i)+str(j)] = float(sim.detach().cpu().numpy())
                            print(i, j, sim)
                    for i in range(0, 3): # 所有梯度清零
                        for name, parms in model.named_parameters():
                            if 'encoder' in name:
                                self.grad_dic['0'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                                self.grad_dic['1'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                                self.grad_dic['2'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)  
                    print(self.epoch, self.args.num_train_epochs)
                    if self.epoch == self.args.num_train_epochs: # 到达最后一个epoch，保存每个epoch的梯度相似度和最后的总梯度
                        sim_file = open(self.sim_file, "w")
                        # grad_file = open(self.grad_file, "wb")
                        i = 0
                        json.dump(self.sim_dic, sim_file)
                        # pickle.dump(self.grad_dic_all, grad_file)
                        sim_file.close()
                        # grad_file.close()
            if self.calculate_gradient_fine_grit:
                for name, parms in model.named_parameters():
                    # print(name, parms, parms.grad)
                    if parms.grad == None:
                        continue
                    self.grad_dic[str(task_index[0])][name] += (parms.grad.clone().detach() - last_step_grad[name])
                    # self.grad_dic_all[str(task_index[0])][name] += parms.grad.clone().detach()
                if (self.step_num / self.args.gradient_accumulation_steps) % self.args.eval_steps == 0: # 过了一个eval_step
                    self.epoch += 1
                    self.sim_dic[str(self.epoch)] = {}
                    for i in range(0, 1): # 计算梯度的余弦相似度
                        for j in range(i + 1, 2):
                            self.sim_dic[str(self.epoch)][str(i)+str(j)] = {}
                            for key, value1 in self.grad_dic[str(i)].items():
                                value2 = self.grad_dic[str(j)][key]
                                tmp1 = torch.sum((value1 * value2).reshape(1, -1))
                                tmp2 = torch.sum((value1 * value1).reshape(1, -1))
                                tmp3 = torch.sum((value2 * value2).reshape(1, -1))
                                sim = tmp1 / (math.sqrt(tmp2) * math.sqrt(tmp3) + 1e-8)
                                if sim != 0.0:
                                    self.sim_dic[str(self.epoch)][str(i)+str(j)][key] = float(sim.detach().cpu().numpy())
                    for i in range(0, 3): # 所有梯度清零
                        for name, parms in model.named_parameters():
                            self.grad_dic['0'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                            self.grad_dic['1'][name] = torch.zeros(parms.shape, dtype=float, device='cuda', requires_grad=False)
                    if self.epoch == self.args.num_train_epochs: # 到达最后一个epoch，保存每个epoch的梯度相似度和最后的总梯度
                        sim_file = open(self.sim_file, "w")
                        i = 0
                        json.dump(self.sim_dic, sim_file)
                        sim_file.close()          
            return loss.detach()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.calculate_gradient or self.multi_decoder:
            train_sampler = None
        else:
            # train_sampler = self._get_train_sampler()
            train_sampler = None

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        out_list = {}
        for task_name, dataset in self.eval_dataset.items():
            self.compute_metrics = self.multi_task_metrics[task_name]
            eval_dataloader = self.get_eval_dataloader(dataset)
            start_time = time.time()
            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self.log(output.metrics)

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

            self._memory_tracker.stop_and_update_metrics(output.metrics)
            for key, value in output.metrics.items():
                out_list[key] = value

        return out_list


def main(): pass


if __name__ == "__main__":
    main()
