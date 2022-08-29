import json
from re import template
from typing import Sequence
import numpy as np

def extract_first_mentions(mentions):
    if mentions == []:
        return ''
    else:
        string = mentions[0][0][0] + ''
        for entity in mentions[1:]:
            string += '<extra_id_2>' + entity[0][0]
        return string

def convert_templates_to_sequence(templates):
    squence = ''
    for template in templates:
        squence += 'incident type: '
        squence += template['incident_type']
        squence += '<extra_id_0>criminal individuals: ' + extract_first_mentions(template['PerpInd'])
        squence += '<extra_id_0>criminal organizations: ' + extract_first_mentions(template['PerpOrg'])
        squence += '<extra_id_0>physical targets: ' + extract_first_mentions(template['Target'])
        squence += '<extra_id_0>human victims: ' + extract_first_mentions(template['Victim'])
        squence += '<extra_id_0>weapons: ' + extract_first_mentions(template['Weapon']) + '<extra_id_1>'
    return squence
# files = ['train.json', 'test.json', 'dev.json']
# for filename in files:
#     f = open(filename)
#     write_file = open('text2event_' + filename, 'w')
#     jsons = f.readlines()
#     for data in jsons:
#         dic = json.loads(data)
#         text = dic['doctext']
#         event = convert_templates_to_sequence(dic['templates'])
#         sample = {'text':text, 'event':event, 'templates':str(dic['templates'])}
#         a = json.dumps(sample)
#         write_file.writelines(a)
#     write_file.close()

def convert_templates_to_event_type(templates):
    squence = ''
    for template in templates:
        squence += template['incident_type'] + '<extra_id_1>'
    return squence

def convert_templates_to_full(templates):
    squence = ''
    for template in templates:
        squence += template['incident_type'] + '<extra_id_1>'
    for template in templates:
        squence += extract_first_mentions(template['PerpInd'])
        squence += '<extra_id_0>' + extract_first_mentions(template['PerpOrg'])
        squence += '<extra_id_0>' + extract_first_mentions(template['Target'])
        squence += '<extra_id_0>' + extract_first_mentions(template['Victim'])
        squence += '<extra_id_0>' + extract_first_mentions(template['Weapon']) + '<extra_id_1>'
    return squence

def convert_templates_to_full2(templates):
    squence = ''
    for template in templates:
        squence += template['incident_type'] + '<extra_id_3>'
        squence += extract_first_mentions(template['PerpInd'])
        squence += '<extra_id_4>' + extract_first_mentions(template['PerpOrg'])
        squence += '<extra_id_5>' + extract_first_mentions(template['Target'])
        squence += '<extra_id_6>' + extract_first_mentions(template['Victim'])
        squence += '<extra_id_7>' + extract_first_mentions(template['Weapon']) + '<extra_id_1>'
    return squence

def my_isalpha(string):
    for i in range(len(string)):
        if string[i].isalpha():
            return True
    return False

def to_text(h, t, r):
    text = ''
    text += r
    text += '<extra_id_1>'
    dic = {'<extra_id_3>':'ORG', '<extra_id_4>':'LOC', '<extra_id_5>':'NULL', '<extra_id_6>':'NUM', '<extra_id_7>':'MISC'}
    lis = [h, t]
    mark = [0, 0]

    if lis[0]['type'] == 'PER':
        text += lis[0]['name']
        mark[0] = 1
        if lis[1]['type'] == 'PER':
            text += '<extra_id_2>' + lis[1]['name']
            mark[1] = 1
    elif lis[1]['type'] == 'PER':
        text += lis[1]['name']
        mark[1] = 1

    for i in range(3, 8):
        token = '<extra_id_' + str(i) + '>'
        text += token
        type = dic[token]
        sign = 0
        for j in range(len(lis)):
            if mark[j] == 1:
                continue
            if lis[j]['type'] == type:
                if sign == 1:
                    text += '<extra_id_2>'
                sign = 1
                text += lis[j]['name']
                mark[j] = 1
    text += '<extra_id_1>'
    return text

def entity_to_text(entitys):
    text = ''
    # dic = {'<extra_id_1>':'PER', '<extra_id_3>':'ORG', '<extra_id_4>':'LOC', '<extra_id_5>':'NULL', '<extra_id_6>':'NUM', '<extra_id_7>':'MISC',  '<extra_id_8>':'TIME'}
    # dic = {'<extra_id_11>':'PER', '<extra_id_12>':'ORG', '<extra_id_13>':'LOC', '<extra_id_14>':'NUM', '<extra_id_15>':'MISC',  '<extra_id_16>':'TIME'}
    dic = {'<extra_id_11>':'PER', '<extra_id_12>':'ORG', '<extra_id_13>':'LOC', '<extra_id_14>':'MISC'}
    mark = np.zeros(len(entitys), dtype=int)
    for key, value in sorted(dic.items(), key=lambda x:x[0]):
        first = True
        text += key
        for i in range(len(entitys)):
            if mark[i] == 0:
                if entitys[i]['type'] == value:
                    if first:
                        first = False
                    else:
                        # text += '<extra_id_2>'
                        text += '<extra_id_17>'
                    text += entitys[i]['name']
                    mark[i] = 1
    print(text)
    return text

def main2():
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
        max_num = 1260
        count = 0
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
            for entity in dic['vertexSet']:
                entity_list.append(entity[0])

            labels += entity_to_text(entity_list)
        
            text = text.replace('\\n', '')
            labels = labels.replace('\\n', '')
            sample = {'text':text, 'event':labels}
            samples.append(sample)
            count += 1
            if count > max_num:
                break
            if count % 5000 == 0:
                print(count)
    write_file = open('NER_1260.json', 'w')

    for sample in samples:
        a = json.dumps(sample)
        write_file.writelines(a)
    write_file.close()

def main():
    f = open('rel_info.json')
    rel_dic = json.loads(f.readlines()[0])
    # print(rel_dic)
    files = ['train_annotated.json']
    types = ['PER', 'LOC', 'ORG', 'MISC']
    samples = []
    for filename in files:
        f = open(filename)
        jsons = f.readlines()
        dics = json.loads(jsons[0])
        print(len(dics))
        max_num = 90000
        count = 0
        relation_count = 0
        doc_count = len(dics)
        for dic in dics:
            text = ''
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
            for i in range(len(dic['labels'])-1):
                for j in range(i, len(dic['labels'])):
                    x1 = dic['labels'][i]['h']
                    x2 = dic['labels'][i]['t']
                    y1 = dic['labels'][j]['h']
                    y2 = dic['labels'][j]['t']
                    if x1 > x2:
                        tmp = x1
                        x1 = x2
                        x2 = tmp
                    if y1 > y2:
                        tmp = y1
                        y1 = y2
                        y2 = tmp
                    if y2 < x2:
                        print(x1, x2, y1, y2)
                        print('True')

            for label in dic['labels']:
                h = entity_list[label['h']]
                t = entity_list[label['t']]
                if h['type'] not in types or t['type'] not in types:
                    continue
                r = rel_dic[label['r']]
                labels += to_text(h, t, r)
                relation_count += 1
            text = text.replace('\\n', '')
            labels = labels.replace('\\n', '')
            sample = {'text':text, 'event':labels}
            samples.append(sample)
            count += 1
            if count > max_num:
                break
            if count % 500 == 0:
                print(count)
        print(relation_count/doc_count)
    write_file = open('relation.json', 'w')

    for sample in samples:
        a = json.dumps(sample)
        write_file.writelines(a)
    write_file.close()

main2()

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
