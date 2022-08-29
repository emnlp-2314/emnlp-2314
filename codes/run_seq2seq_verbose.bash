#!/usr/bin/env bash
# -*- coding:utf-8 -*-

EXP_ID=$(date +%F-%H-%M-$RANDOM)
export CUDA_VISIBLE_DEVICES="0"
export batch_size="4"
export model_name='./t5-base'

export file_name='fine_grid_NER_3'
export cut_reverse=0
export share_num=12
export data_name=one_ie_ace2005_subtype
export lr=1e-4
export task_name="event"
export seed="3"
export lr_scheduler='linear' 
export label_smoothing="0"
export epoch=50
export decoding_format='myparser'
export eval_steps=272
export warmup_steps=272
export constraint_decoding=''
OPTS=$(getopt -o b:d:m:i:t:s:l:f: --long batch:,device:,model:,data:,task:,seed:,lr:,lr_scheduler:,label_smoothing:,epoch:,format:,eval_steps:,warmup_steps:,wo_constraint_decoding -n 'parse-options' -- "$@")

if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
  -b | --batch)
    batch_size="$2"
    shift
    shift
    ;;
  -d | --device)
    CUDA_VISIBLE_DEVICES="$2"
    shift
    shift
    ;;
  -m | --model)
    model_name="$2"
    shift
    shift
    ;;
  -i | --data)
    data_name="$2"
    shift
    shift
    ;;
  -t | --task)
    task_name="$2"
    shift
    shift
    ;;
  -s | --seed)
    seed="$2"
    shift
    shift
    ;;
  -l | --lr)
    lr="$2"
    shift
    shift
    ;;
  -f | --format)
    decoding_format="$2"
    shift
    shift
    ;;
  --lr_scheduler)
    lr_scheduler="$2"
    shift
    shift
    ;;
  --label_smoothing)
    label_smoothing="$2"
    shift
    shift
    ;;
  --epoch)
    epoch="$2"
    shift
    shift
    ;;
  --eval_steps)
    eval_steps="$2"
    shift
    shift
    ;;
  --warmup_steps)
    warmup_steps="$2"
    shift
    shift
    ;;
  --wo_constraint_decoding)
    constraint_decoding=""
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "$1" not recognize.
    exit
    ;;
  esac
done

# google/mt5-base -> google_mt5-base
model_name_log=$(echo ${model_name} | sed -s "s/\//_/g")

model_folder=models/CF_share_num_${share_num}_seed_${seed}_cut_reverse_${cut_reverse}
data_folder=data/text2tree/${data_name}

export TOKENIZERS_PARALLELISM=false
gradient_accumulation_steps=4
output_dir=${model_folder}

# export eval_steps=81
# export warmup_steps=81
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_seq2seq.py \
#     --file_name=${file_name} \
#     --num_beams=1 \
#     --do_train --do_eval --do_predict ${constraint_decoding} \
#     --label_smoothing_sum=False \
#     --use_fast_tokenizer=False \
#     --save_steps ${eval_steps} \
#     --evaluation_strategy steps \
#     --predict_with_generate \
#     --metric_for_best_model eval_micro_avg_f1 \
#     --save_total_limit 1 \
#     --load_best_model_at_end \
#     --max_source_length=512 \
#     --max_target_length=100 \
#     --num_train_epochs=${epoch} \
#     --task=${task_name} \
#     --train_file=${data_folder}/muti-task-train.json \
#     --validation_file=${data_folder}/muti-task-dev.json \
#     --test_file=${data_folder}/muti-task-test.json \
#     --event_schema=${data_folder}/event.schema \
#     --per_device_train_batch_size=${batch_size} \
#     --per_device_eval_batch_size=$((batch_size * 4)) \
#     --output_dir=${output_dir} \
#     --logging_dir=${output_dir}_log \
#     --model_name_or_path=${model_name} \
#     --learning_rate=${lr} \
#     --lr_scheduler_type=${lr_scheduler} \
#     --label_smoothing_factor=${label_smoothing} \
#     --eval_steps ${eval_steps} \
#     --decoding_format ${decoding_format} \
#     --warmup_steps ${warmup_steps} \
#     --source_prefix="${task_name}: " \
#     --seed=${seed} \
#     --gradient_accumulation_steps=${gradient_accumulation_steps}

# export eval_steps=272
# export warmup_steps=272
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_seq2seq.py \
#     --file_name=${file_name} \
#     --num_beams=1 \
#     --do_train --do_eval --do_predict ${constraint_decoding} \
#     --label_smoothing_sum=False \
#     --use_fast_tokenizer=False \
#     --save_steps ${eval_steps} \
#     --evaluation_strategy steps \
#     --predict_with_generate \
#     --metric_for_best_model eval_micro_avg_f1 \
#     --save_total_limit 1 \
#     --load_best_model_at_end \
#     --max_source_length=512 \
#     --max_target_length=100 \
#     --num_train_epochs=${epoch} \
#     --task=${task_name} \
#     --train_file=${data_folder}/mixed_train.json \
#     --validation_file=${data_folder}/muti-task-dev.json \
#     --test_file=${data_folder}/muti-task-test.json \
#     --event_schema=${data_folder}/event.schema \
#     --per_device_train_batch_size=${batch_size} \
#     --per_device_eval_batch_size=$((batch_size * 4)) \
#     --output_dir=${output_dir} \
#     --logging_dir=${output_dir}_log \
#     --model_name_or_path=${model_name} \
#     --learning_rate=${lr} \
#     --lr_scheduler_type=${lr_scheduler} \
#     --label_smoothing_factor=${label_smoothing} \
#     --eval_steps ${eval_steps} \
#     --decoding_format ${decoding_format} \
#     --warmup_steps ${warmup_steps} \
#     --source_prefix="${task_name}: " \
#     --seed=${seed} \
#     --gradient_accumulation_steps=${gradient_accumulation_steps}
export batch_size="4"
gradient_accumulation_steps=4
export eval_steps=163
export warmup_steps=163
epoch=50
export NER=0
export RE=0
export both=1
CUDA_VISIBLE_DEVICES="2"
cut_reverse=0
export file_name='fine_grid_CoreNLP'
export fine_grit_share_parm=1
export train_file='CoreNLP_1300_NER_656_RE_652_.json'
for seed in "0" "1" "2" "3"
  do
  for share_num in 15
    do
    file_name=fine_grid_all_${seed}
    model_folder=models/TEST_${share_num}_${fine_grit_share_parm}_${train_file}_${seed}
    output_dir=${model_folder}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_seq2seq.py \
        --file_name=${file_name} \
        --cut_reverse=${cut_reverse} \
        --share_num=${share_num} \
        --num_beams=1 \
        --do_train --do_eval --do_predict${constraint_decoding} \
        --label_smoothing_sum=False \
        --use_fast_tokenizer=False \
        --save_steps ${eval_steps} \
        --evaluation_strategy steps \
        --predict_with_generate \
        --metric_for_best_model eval_micro_avg_f1 \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --max_source_length=512 \
        --max_target_length=256 \
        --num_train_epochs=${epoch} \
        --task=${task_name} \
        --train_file=${data_folder}/${train_file} \
        --validation_file=${data_folder}/muti-task-dev.json \
        --test_file=${data_folder}/muti-task-test.json \
        --event_schema=${data_folder}/event.schema \
        --per_device_train_batch_size=${batch_size} \
        --per_device_eval_batch_size=$((batch_size * 4)) \
        --output_dir=${output_dir} \
        --logging_dir=${output_dir}_log \
        --model_name_or_path=${model_name} \
        --learning_rate=${lr} \
        --lr_scheduler_type=${lr_scheduler} \
        --label_smoothing_factor=${label_smoothing} \
        --eval_steps ${eval_steps} \
        --decoding_format ${decoding_format} \
        --warmup_steps ${warmup_steps} \
        --source_prefix="${task_name}: " \
        --seed=${seed} \
        --gradient_accumulation_steps=${gradient_accumulation_steps} \
        --fine_grit_share_parm=${fine_grit_share_parm} \
        --NER=${NER} \
        --RE=${RE} \
        --both=${both} \
        --loss_dir=${output_dir}
    done
  done