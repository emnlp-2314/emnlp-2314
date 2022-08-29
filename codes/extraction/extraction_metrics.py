from typing import List
from copy import deepcopy
from extraction.event_schema import EventSchema
from extraction.predict_parser.predict_parser import Metric
from extraction.predict_parser.tree_predict_parser import MyPredictParser, NERParser
import numpy as np
import re
import string
import json
import argparse
from collections import OrderedDict
import itertools
import copy
tag2role = OrderedDict({'incident_type':'incident_type', 'perp_individual_id': "PerpInd", 'perp_organization_id': "PerpOrg", 'phys_tgt_id': "Target", 'hum_tgt_name': "Victim", 'incident_instrument_id': "Weapon"})

decoding_format_dict = {
    'myparser': MyPredictParser,
    'NER': NERParser
}


def get_predict_parser(format_name):
    return decoding_format_dict[format_name]


def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


def matching(c1, c2):
    # similarity: if c2 (pred) is subset of c1 (gold) return 1 
    for m in c2:
        if m not in c1:
            return 0
    return 1

def is_valid_mapping(mapping):
    reverse_mapping = {}
    for k in mapping:
        v = mapping[k]
        if v not in reverse_mapping:
            reverse_mapping[v] = [k]
        else:
            reverse_mapping[v].append(k)

    for v in reverse_mapping:
        if v == -1: continue
        if len(reverse_mapping[v]) > 1:
            return False

    return True
all_count = 0

def weak_recall(pred, gold):
    pred_entitys = []
    gold_entitys = []
    for event in pred:
        for key, roles in event.items():
            if key == "incident_type":
                continue
            for role in roles:
                if role not in pred_entitys:
                    pred_entitys.append(role)
    for event in gold:
        for key, roles in event.items():
            if key == "incident_type":
                continue
            for role in roles:
                if role not in gold_entitys:
                    gold_entitys.append(role)
    
    weak_recall_num = 0
    for role in pred_entitys:
        for gold_role in gold_entitys:
            if role[0] in gold_role:
                weak_recall_num += 1
                break
    
    return weak_recall_num, len(gold_entitys)

def weak_relation(pred, gold):
    hit_count = 0
    relation_count = 0
    pred_entitys = []
    gold_relations = []
    pred_relations = []
    for event in pred:
        for key, roles in event.items():
            if key == "incident_type":
                continue
            for role in roles:
                if role not in pred_entitys:
                    pred_entitys.append(role)
    # print(pred_entitys)
    for event in gold:
        entities_tmp = []
        for key1, roles1 in event.items():
            if key1 == "incident_type":
                continue
            for entitiy in roles1:
                entities_tmp.append(entitiy)
        # print(entities_tmp)
        for i in range(0, len(entities_tmp) - 1 ):
            for j in range(i + 1, len(entities_tmp) ):
                entity1 = entities_tmp[i]
                entity2 = entities_tmp[j]
                hit1 = 0
                hit2 = 0
                for entity in pred_entitys:
                    if entity[0] in entity1:
                        hit1 = 1
                    if entity[0] in entity2:
                        hit2 = 1
                if hit1 == 1 and hit2 == 1:
                    gold_relations.append((entity1, entity2))
    relation_count = len(gold_relations)

    for event in pred:
        entities_tmp = []
        for key1, roles1 in event.items():
            if key1 == "incident_type":
                continue
            for entitiy in roles1:
                entities_tmp.append(entitiy)
        
        for i in range(0, len(entities_tmp) - 1 ):
            for j in range(i + 1, len(entities_tmp) ):
                entity1 = entities_tmp[i]
                entity2 = entities_tmp[j]
                pred_relations.append((entity1, entity2))

    for pred_relation in pred_relations:
        # print(pred_relation)
        # print(gold_relations)
        for i in range(0, len(gold_relations)):
            gold_relation = gold_relations[i]
            if pred_relation[0][0] in gold_relation[0] and pred_relation[1][0] in gold_relation[1]:
                hit_count += 1
                del gold_relations[i]
                break
            if pred_relation[1][0] in gold_relation[0] and pred_relation[0][0] in gold_relation[1]:
                hit_count += 1
                del gold_relations[i]
                break
    
    return hit_count, relation_count

def strict_score(mapping, pred, gold, f=False):
    ex_result = OrderedDict()
    all_keys = list(role for _, role in tag2role.items()) + ["micro_avg"]
    for key in all_keys:
        ex_result[key] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}
    # if invalid mapping, return 0
    # if not is_valid_mapping(mapping):
    #     return ex_result
    global all_count
    mapped_temp_pred = []
    mapped_temp_gold = []
    for pred_temp_idx in mapping:
        gold_temp_idx = mapping[pred_temp_idx]
        if type(pred[pred_temp_idx]["incident_type"]) != str:
            pred[pred_temp_idx]["incident_type"] = "attack"
        if gold_temp_idx != -1 and pred[pred_temp_idx]["incident_type"] in gold[gold_temp_idx]["incident_type"]: # attach vs attach / bombing
            mapped_temp_pred.append(pred_temp_idx)
            mapped_temp_gold.append(gold_temp_idx)
            pred_temp, gold_temp = pred[pred_temp_idx], gold[gold_temp_idx]
            
            # prec
            for role in pred_temp.keys():
                if role == "incident_type":
                    ex_result[role]["p_den"] += 1
                    ex_result[role]["p_num"] += 1
                    continue

                for entity_pred in pred_temp[role]:
                    ex_result[role]["p_den"] += 1
                    correct = False
                    for entity_gold in gold_temp[role]:
                        # import ipdb; ipdb.set_trace()
                        if matching(entity_gold, entity_pred):
                            correct = True
                    if correct:
                        ex_result[role]["p_num"] += 1

            # recall
            for role in gold_temp.keys():
                if role == "incident_type": 
                    ex_result[role]["r_den"] += 1
                    ex_result[role]["r_num"] += 1
                    continue
                
                for entity_gold in gold_temp[role]:
                    ex_result[role]["r_den"] += 1
                    correct = False
                    for entity_pred in pred_temp[role]:
                        if matching(entity_gold, entity_pred):
                            correct = True
                    if correct:
                        ex_result[role]["r_num"] += 1                        
    # spurious
    for pred_temp_idx in range(len(pred)):
        pred_temp = pred[pred_temp_idx]
        if pred_temp_idx not in mapped_temp_pred:
            for role in pred_temp:
                if role == "incident_type": 
                    ex_result[role]["p_den"] += 1
                    continue
                for entity_pred in pred_temp[role]:
                    ex_result[role]["p_den"] += 1

    # missing 
    for gold_temp_idx in range(len(gold)):
        gold_temp = gold[gold_temp_idx]
        if gold_temp_idx not in mapped_temp_gold:
            for role in gold_temp:
                if role == "incident_type": 
                    ex_result[role]["r_den"] += 1
                    continue
                for entity_gold in gold_temp[role]:
                    ex_result[role]["r_den"] += 1

    ex_result["micro_avg"]["p_num"] = sum(ex_result[role]["p_num"] for _, role in tag2role.items())
    ex_result["micro_avg"]["p_den"] = sum(ex_result[role]["p_den"] for _, role in tag2role.items())
    ex_result["micro_avg"]["r_num"] = sum(ex_result[role]["r_num"] for _, role in tag2role.items())
    ex_result["micro_avg"]["r_den"] = sum(ex_result[role]["r_den"] for _, role in tag2role.items())

    for key in all_keys:
        ex_result[key]["p"] = 0 if ex_result[key]["p_num"] == 0 else ex_result[key]["p_num"] / float(ex_result[key]["p_den"])
        ex_result[key]["r"] = 0 if ex_result[key]["r_num"] == 0 else ex_result[key]["r_num"] / float(ex_result[key]["r_den"])
        ex_result[key]["f1"] = f1(ex_result[key]["p_num"], ex_result[key]["p_den"], ex_result[key]["r_num"], ex_result[key]["r_den"])

    return ex_result


def eval_tf_strict(preds, golds):
    # normalization mention strings
    for docid in range(len(preds)):
        for idx_temp in range(len(preds[docid])):
            for role in preds[docid][idx_temp]:
                if role == "incident_type": continue
                for idx in range(len(preds[docid][idx_temp][role])):
                    a = normalize_string(preds[docid][idx_temp][role][idx])
                    preds[docid][idx_temp][role][idx] = [a]
    for docid in range(len(golds)):
        for idx_temp in range(len(golds[docid])):
            for role in golds[docid][idx_temp]:
                if role == "incident_type": continue
                for idx in range(len(golds[docid][idx_temp][role])):
                    for idy in range(len(golds[docid][idx_temp][role][idx])):
                        golds[docid][idx_temp][role][idx][idy] = normalize_string(golds[docid][idx_temp][role][idx][idy][0])


    # get eval results
    result = OrderedDict()
    all_keys = list(role for _, role in tag2role.items()) + ["micro_avg"]
    for key in all_keys:
        b = ["p_num", "p_den", "r_num", "r_den", "p", "r", "f1"]
        for c in b:
            result[key+'_'+c] = 0
    result["weak_recall_num"] = 0
    result["weak_total"] = 0
    result["relation_hit_num"] = 0
    result["relation_total"] = 0
    file = open('out.json', 'w+')
    a = {}

    for docid in range(len(preds)):
        
        pred = preds[docid]
        if len(pred) > 4:
            pred = pred[0:4]
        gold = golds[docid]
        K, V = list(range(len(pred))), list(range(len(gold)))
        V =  V + [-1]
        init_maps = [dict(zip(K, p)) for p in itertools.product(V, repeat=len(K))]
        ex_best = None
        map_best = None
        for mapping in  init_maps:
            if not is_valid_mapping(mapping):
                continue
            ex_result = strict_score(mapping, pred, gold)
            if ex_best is None:
                ex_best = ex_result
                map_best = mapping
            elif ex_result["micro_avg"]["f1"] > ex_best["micro_avg"]["f1"]:
                ex_best = ex_result
                map_best = mapping
        hit_num, total_relaitons = weak_relation(pred, gold)
        result["relation_hit_num"] += hit_num
        result["relation_total"] += total_relaitons
        weak_recall_num, weak_total = weak_recall(pred, gold)
        result["weak_recall_num"] += weak_recall_num
        result["weak_total"] += weak_total
        if len(pred) == 1 and len(gold) == 1:
            a[docid] = {"pred":preds[docid], "gold":golds[docid], "score":ex_best}    
        # sum for one docid
        for role in all_keys:
            if role == "micro_avg": continue
            result[role+'_'+"p_num"] += ex_best[role]["p_num"]
            result[role+'_'+"p_den"] += ex_best[role]["p_den"]
            result[role+'_'+"r_num"] += ex_best[role]["r_num"]
            result[role+'_'+"r_den"] += ex_best[role]["r_den"]
    
    
    # micro average
    result["micro_avg"+'_'+"p_num"] = sum(result[role+'_'+"p_num"] for _, role in tag2role.items())
    result["micro_avg"+'_'+"p_den"] = sum(result[role+'_'+"p_den"] for _, role in tag2role.items())
    result["micro_avg"+'_'+"r_num"] = sum(result[role+'_'+"r_num"] for _, role in tag2role.items())
    result["micro_avg"+'_'+"r_den"] = sum(result[role+'_'+"r_den"] for _, role in tag2role.items())
    
    json.dump(a, file, indent=2)
    file.close()
    
    for key in all_keys:
        result[key+'_'+"p"] = 0 if result[key+'_'+"p_num"] == 0 else result[key+'_'+"p_num"] / float(result[key+'_'+"p_den"])
        result[key+'_'+"r"] = 0 if result[key+'_'+"r_num"] == 0 else result[key+'_'+"r_num"] / float(result[key+'_'+"r_den"])
        result[key+'_'+"f1"] = f1(result[key+'_'+"p_num"], result[key+'_'+"p_den"], result[key+'_'+"r_num"], result[key+'_'+"r_den"])

    result["weak_recall_rate"] = result["weak_recall_num"] / float(result["weak_total"])
    if result["relation_total"] == 0:
        print("jian?")
        result["successful_distribution_rate"] = 0
    else:
        result["successful_distribution_rate"] = result["relation_hit_num"] / float(result["relation_total"])
    return result


def eval_pred(predict_parser, pred_list, template_file=None):
    pred = predict_parser.decode(pred_list)
    if template_file == None: # dev
        file = open("./data/text2tree/one_ie_ace2005_subtype/dev copy.json")
    else:
        file = open(template_file)
    a = []
    for line in file.readlines():
        b = json.loads(line)
        a.append(b["templates"])
    
    result = eval_tf_strict(pred, a)
    return result

def eval_NER(pred, a):
    def has_special_token(text):
        if '<extra_id_61>' in text or '<extra_id_62>' in text or '<extra_id_63>' in text or '<extra_id_64>' in text:
            return True
        else:
            return False
    def get_spans(text):
        dic = {'<extra_id_61>': 1, '<extra_id_62>': 2, '<extra_id_63>': 3, '<extra_id_64>': 4}
        ids = ['<extra_id_61', '<extra_id_62', '<extra_id_63', '<extra_id_64']
        spans = []
        # print(text)
        # len('<extra_id_61>') = 13
        while has_special_token(text):
            begin_idx = 99999
            for id in ids:
                tmp = text.find(id)
                if tmp != -1:
                    if begin_idx > tmp:
                        begin_idx = tmp
            # print(text)
            # print('---', text[begin_idx:begin_idx + 13], begin_idx)
            cls1 = dic[text[begin_idx:begin_idx + 13]]
            end_idx = 99999
            for id in ids:
                tmp = text.find(id, begin_idx + 10)
                if tmp != -1:
                    if end_idx > tmp:
                        end_idx = tmp
            # print('???', text[begin_idx + 13: end_idx], begin_idx + 13, end_idx)
            text = text[0: begin_idx] + text[begin_idx + 13: end_idx] + text[end_idx + 13:]
            spans.append({'begin': begin_idx, 'end': end_idx, 'class': cls1})
        return spans
    hit = 0
    false = 0
    total_label = 0
    for i in range(len(pred)):
        pred_text = pred[i]
        ground_truth = a[i]
        pred_spans = get_spans(pred_text)
        label_spans = get_spans(ground_truth)
        if pred_spans != []:
            for pred_span in pred_spans:
                pred_span_1 = copy.deepcopy(pred_span)
                pred_span_1['begin'] += 1
                pred_span_2 = copy.deepcopy(pred_span_1)
                pred_span_2['begin'] += 1
                pred_span_2['end'] += 1
                pred_span_3 = copy.deepcopy(pred_span_2)
                pred_span_3['begin'] += 1
                pred_span_3['end'] += 1
                pred_span_4 = copy.deepcopy(pred_span_3)
                pred_span_4['begin'] += 1
                pred_span_4['end'] += 1
                pred_span_5 = copy.deepcopy(pred_span_4)
                pred_span_5['begin'] += 1
                pred_span_5['end'] += 1
                pred_span_6 = copy.deepcopy(pred_span_1)
                pred_span_6['begin'] -= 1
                pred_span_6['end'] -= 1
                pred_span_7 = copy.deepcopy(pred_span_6)
                pred_span_7['begin'] -= 1
                pred_span_7['end'] -= 1
                pred_span_8 = copy.deepcopy(pred_span_7)
                pred_span_8['begin'] -= 1
                pred_span_8['end'] -= 1
                pred_span_9 = copy.deepcopy(pred_span_8)
                pred_span_9['begin'] -= 1
                pred_span_9['end'] -= 1
                pred_span_10 = copy.deepcopy(pred_span_9)
                pred_span_10['begin'] -= 1
                pred_span_10['end'] -= 1
                if pred_span in label_spans or pred_span_1 in label_spans or pred_span_2 in label_spans or pred_span_3 in label_spans or pred_span_4 in label_spans or pred_span_5 in label_spans or pred_span_6 in label_spans or pred_span_7 in label_spans or pred_span_8 in label_spans or pred_span_9 in label_spans or pred_span_10 in label_spans:
                    hit += 1
                else: 
                    false += 1
        total_label += len(label_spans)
    result = {}
    result['precision'] = hit / (hit + false + 1e-8)
    result['recall'] = hit / (total_label + 1e-8)
    result['micro_avg_f1'] = 2 * result['precision'] * result['recall'] / (result['precision'] + result['recall'] + 1e-8)
    return result

def eval_NER_pred(predict_parser, pred_list, template_file=None):
    pred = predict_parser.decode(pred_list)
    if template_file == None: # dev
        def get_jsons(filename):
            d = json.JSONDecoder()
            x = open(filename,'r').read()
            jlist=[]
            while True:
                try:
                    j,n = d.raw_decode(x) 
                    jlist.append(j)
                except ValueError: 
                    break
                x=x[n:]
            return jlist
        dics = get_jsons("./data/text2tree/one_ie_ace2005_subtype/CoNLL03_valid.json")
    
    a = []
    for dic in dics:
        a.append(dic['templates'])
    result = eval_NER(pred, a)
    return result

def get_extract_metrics(pred_lns: List[str], decoding_format='tree', template_file=None):
    predict_parser = get_predict_parser(format_name=decoding_format)()
    if decoding_format=='NER':
        return eval_NER_pred(
            predict_parser,
            pred_lns,
            template_file
        )
    else:
        return eval_pred(
            predict_parser,
            pred_lns,
            template_file
        )

with open('/casestudy/good.txt') as good:
    lines = good.readlines()
    print(get_extract_metrics(list(lines), template_file='/casestudy/test.json'))