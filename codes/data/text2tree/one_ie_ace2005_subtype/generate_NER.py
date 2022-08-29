# -*- coding:utf-8 -*-
import json
from re import template
from typing import Sequence
import numpy as np
import random


def main():
    f = open('rel_info.json')
    rel_dic = json.loads(f.readlines()[0])
    # print(rel_dic)
    files = ['train_annotated.json']
    samples = []
    for filename in files:
        f = open(filename)
        jsons = f.readlines()
        dics = json.loads(jsons[0])
        print(len(dics))
        max_num = 5000
        count = 0
        relation_count = 0
        doc_count = len(dics)
        for dic in dics:
            text = '<extra_id_31>'
            labels = ''
            for sent in dic['sents']:
                double_stack = 0
                single_stack = 0
                for i in range(len(sent) - 1):
                    if sent[i+1] == '\"' and double_stack == 0:
                        double_stack = 1   
                    elif sent[i+1] == '\"' and double_stack == 1:
                        double_stack = 0
                    if sent[i+1] == '\'' and single_stack == 0:
                        single_stack = 1   
                    elif sent[i+1] == '\'' and single_stack == 1:
                        single_stack = 0
                    if sent[i] in ['(', '/', ":", '-', '–', '-', '_', '\'']:
                        text += sent[i]
                    elif sent[i] in ['\"'] and double_stack == 1 and (sent[i+1].isalpha() or sent[i+1].isdigit()):
                        text += sent[i]  
                    elif sent[i] in ['\''] and single_stack == 1 and (sent[i+1].isalpha() or sent[i+1].isdigit()):
                        text += sent[i] 
                    elif sent[i+1].isalpha() or sent[i+1].isdigit():
                        text += sent[i] + ' '
                    elif sent[i+1] in ['\"'] and double_stack == 1:
                        text += sent[i] + ' '
                    elif sent[i+1] in ['\"'] and double_stack == 0:
                        text += sent[i]
                    elif sent[i+1] in ['\''] and single_stack == 1:
                        text += sent[i] + ' '
                    elif sent[i+1] in ['\''] and single_stack == 0:
                        text += sent[i]
                    else:
                        text += sent[i]
                text += sent[-1] + ' '
            entity_list = []
            for entitys in dic['vertexSet']:
                for entity in entitys:
                    entity_list.append(entity)

            def random_drop(entity_list):
                num = round(len(entity_list) * 0.5)
                count = 0
                while count < num:
                    count += 1
                    if len(entity_list) == 0:
                        break
                    idx = random.randint(0, len(entity_list) - 1)
                    del entity_list[idx]
            
            random_drop(entity_list)
            entity_list = sorted(entity_list, key=lambda x:(x['sent_id']) * 1000 + x['pos'][0])
            for entity in entity_list:
                labels += entity['name']
                labels += '<extra_id_40>'
            text = text.replace('\\n', '')
            sample = {'text':text, 'event':labels}
            samples.append(sample)
            count += 1
            if count > max_num:
                print("done")
                break
            if count % 500 == 0:
                print(count)
        print(relation_count/doc_count)
    write_file = open('NER_NEW.json', 'w')

    for sample in samples:
        a = json.dumps(sample)
        write_file.writelines(a)
    write_file.close()

main()

# files = ['train_annotated.json']
# for filename in files:
#     f = open(filename)
#     jsons = f.readlines()
#     dics = json.loads(jsons[0])
#     lis = []
#     for dic in dics:
#         for entity in dic['vertexSet']:
#             if entity[0]['type'] not in lis:
#                 lis.append(entity[0]['type'])
    
#     print(lis)
