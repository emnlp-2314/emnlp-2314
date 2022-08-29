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
    types = ['PER', 'LOC', 'ORG', 'MISC']
    map = {'PER':'<extra_id_52>', 'ORG':'<extra_id_53>', 'LOC':'<extra_id_54>', 'MISC':'<extra_id_55>'}
    samples = []
    for filename in files:
        f = open(filename)
        jsons = f.readlines()
        dics = json.loads(jsons[0])
        print(len(dics))
        max_num = 1260
        count = 0
        relation_count = 0
        doc_count = len(dics)
        for dic in dics:
            text = '<extra_id_33>'
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
            for entity in dic['vertexSet']:
                entity_list.append(entity[0])
            # for i in range(len(dic['labels'])-1):
            #     for j in range(i, len(dic['labels'])):
            #         x1 = dic['labels'][i]['h']
            #         x2 = dic['labels'][i]['t']
            #         y1 = dic['labels'][j]['h']
            #         y2 = dic['labels'][j]['t']
            #         if x1 > x2:
            #             tmp = x1
            #             x1 = x2
            #             x2 = tmp
            #         if y1 > y2:
            #             tmp = y1
            #             y1 = y2
            #             y2 = tmp
            #         if y2 < x2:
            #             print(x1, x2, y1, y2)
            #             print('True')


            relations = []
            for label in dic['labels']:
                h = entity_list[label['h']]
                t = entity_list[label['t']]
                if h['type'] not in types or t['type'] not in types:
                    continue
                tup = [map[h['type']] + h['name'] + map[t['type']] + t['name'], '<extra_id_50>']
                tup1 = [map[t['type']] + t['name'] + map[h['type']] + h['name'], '<extra_id_50>']
                if tup in relations or tup1 in relations:
                    continue
                relations.append(tup)

            def is_a_relation(x, y):
                for label in dic['labels']:
                    if label['h'] == x and label['t'] == y:
                        return True
                    if label['h'] == y and label['t'] == x:
                        return True
                return False

            negs = []
            for i in range(len(entity_list) - 1):
                for j in range(i + 1, len(entity_list)):
                    if not is_a_relation(i, j):
                        h = entity_list[i]
                        t = entity_list[j]
                        if h['type'] not in types or t['type'] not in types:
                            continue
                        negs.append([map[h['type']] + h['name'] + map[t['type']] + t['name'], '<extra_id_51>'])
            
            for i in range(0, len(negs)):
                j = random.randint(i, len(negs) - 1)
                tmp = negs[i]
                negs[i] = negs[j]
                negs[j] = tmp

            if len(negs) > len(relations):
                negs = negs[0:len(relations)]
            relations = relations + negs
            
            def add_noise(relations):
                length = len(relations)
                num = round(length * 0.5)
                lis = []
                while len(lis) < num:
                    a = random.randint(0, length - 1)
                    if a in lis:
                        continue
                    else:
                        lis.append(a)
                i = 0
                for relation in relations:
                    if i in lis:
                        print(i)
                        if relation[1] == '<extra_id_51>':
                            relation[1] = '<extra_id_50>'
                        else:
                            relation[1] = '<extra_id_51>'
                    i += 1
            def same(relations):
                for relation in relations:
                    if relation[1] == '<extra_id_51>':
                        relation[1] = '<extra_id_50>'

            # add_noise(relations)
            
            for i in range(0, len(relations)):
                j = random.randint(i, len(relations) - 1)
                tmp = relations[i]
                relations[i] = relations[j]
                relations[j] = tmp
            for i in range(0, len(relations)):
                text += relations[i][0]
                labels += relations[i][0] + relations[i][1]

            text = text.replace('\\n', '')
            labels = labels.replace('\\n', '')
            if len(labels) < 1100:
                continue
            sample = {'text':text, 'event':labels}
            samples.append(sample)
            count += 1
            if count > max_num:
                print("done")
                break
            if count % 500 == 0:
                print(count)
        print(relation_count/doc_count)
    write_file = open('relation_jugement_1260_min_len_1100.json', 'w')

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
