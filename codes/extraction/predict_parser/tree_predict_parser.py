from collections import Counter
from typing import Tuple, List, Dict

from nltk.tree import ParentedTree
import re
import json
from extraction.predict_parser.predict_parser import PredictParser

def role_separator(i):
    return '<extra_id_' + str(i + 3) + '>'
    # return '<extra_id_' + str(0) + '>'

def role_separator2(i):
    return '<extra_id_' + str(i + 4) + '>'
    # return '<extra_id_' + str(0) + '>'

event_separator = '<extra_id_1>'
entity_separator = '<extra_id_2>'
entity_sign = '▏'
role_sign = '▌'
event_sign = '▉'

class MyPredictParser2(PredictParser):

    def decode(self, pred_list):
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
            Counter:
        """

        def convert_extra_id_to_sign(_text):
            _text = _text.replace(' ' + entity_separator + ' ', entity_sign)
            _text = _text.replace(' ' + event_separator + ' ', event_sign)
            _text = _text.replace(entity_separator + ' ', entity_sign)
            _text = _text.replace(event_separator + ' ', event_sign)
            _text = _text.replace(' ' + entity_separator, entity_sign)
            _text = _text.replace(' ' + event_separator, event_sign)
            _text = _text.replace(entity_separator, entity_sign)
            _text = _text.replace(event_separator, event_sign)
            return _text
        def is_empty(dic):
            a = [item[1] for item in dic.items()]
            for a in a:
                if a != '' and a != []:
                    return 0
            return 1
        def delete_space(entitys):
            new = []
            for entity in entitys:
                if entity[0] == ' ':
                    entity = entity[1:]
                if entity[-1] == ' ':
                    entity = entity[0:-1]
                if entity not in new:
                    new.append(entity)
            return new

        map_list = ['PerpInd', 'PerpOrg', 'Target', 'Victim', 'Weapon']
        event_type = ['attack', 'bombing', 'kidnapping', 'arson', 'robbery', 'forced work stoppage']
        out_list = []
        for pred in pred_list:
            # print(pred)
            event_bucket = convert_extra_id_to_sign(pred).split(event_sign)
            # print(event_bucket)
            event_list = []
            event_count = 0
            for i in range(len(event_bucket)):
                if event_bucket[i] in event_type:
                    event_list.append({'incident_type': event_bucket[i]})
                    event_count += 1
            for i in range(event_count):
                for m in map_list:
                    event_list[i][m] = []
            for i in range(event_count, event_count * 2):
                try:
                    role_str = event_bucket[i]
                    
                    for j in range(0, 5):
                        if j == 4:
                            try:
                                entity_list = role_str.split(entity_sign)
                                if entity_list == [''] or entity_list ==[' '] or entity_list == ['  ']:
                                    event_list[i - event_count][map_list[4]] = []
                                else:
                                    event_list[i - event_count][map_list[4]] = delete_space(entity_list)
                            except:
                                pass
                        else:
                            try:
                                role_bucket = role_str.split(role_separator(j))[0]
                                role_str = role_str.split(role_separator(j))[1]
                                entity_list = role_bucket.split(entity_sign)
                                if entity_list == [''] or entity_list == [' '] or entity_list == ['  ']:
                                    event_list[i - event_count][map_list[j]] = []
                                else:
                                    event_list[i - event_count][map_list[j]] = delete_space(entity_list)
                            except:
                                pass

                except:
                    pass
            out_list.append(event_list)
        # print(out_list)
        r = open('./data/text2tree/one_ie_ace2005_subtype/test copy.json')
        w = open('./data/text2tree/one_ie_ace2005_subtype/preds_gtt.out', 'w')
        out_dic = {}
        count = 0
        for line in r.readlines():
            line = json.loads(line)
            docid = str(int(line['docid'].split("-")[0][-1])*10000 + int(line['docid'].split("-")[-1]))
            out_dic[docid] = {}
            out_dic[docid]['pred_templates'] = out_list[count]
            count += 1
        json.dump(out_dic, w)
        w.close()

        return out_list


class MyPredictParser(PredictParser):
    # for not optimized template 
    def decode(self, pred_list):
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
            Counter:
        """

        def convert_extra_id_to_sign(_text):
            _text = _text.replace(' ' + entity_separator + ' ', entity_sign)
            _text = _text.replace(' ' + event_separator + ' ', event_sign)
            _text = _text.replace(entity_separator + ' ', entity_sign)
            _text = _text.replace(event_separator + ' ', event_sign)
            _text = _text.replace(' ' + entity_separator, entity_sign)
            _text = _text.replace(' ' + event_separator, event_sign)
            _text = _text.replace(entity_separator, entity_sign)
            _text = _text.replace(event_separator, event_sign)
            return _text
        def is_empty(dic):
            a = [item[1] for item in dic.items()]
            for a in a:
                if a != '' and a != []:
                    return 0
            return 1
        def delete_space(entitys):
            new = []
            for entity in entitys:
                if entity[0] == ' ':
                    entity = entity[1:]
                if entity[-1] == ' ':
                    entity = entity[0:-1]
                if entity not in new:
                    new.append(entity)
            return new

        map_list = ['PerpInd', 'PerpOrg', 'Target', 'Victim', 'Weapon']
        event_type = ['attack', 'bombing', 'kidnapping', 'arson', 'robbery', 'forced work stoppage']
        out_list = []
        for pred in pred_list:
            # print(pred)
            event_bucket = convert_extra_id_to_sign(pred).split(event_sign)
            # print(event_bucket)
            event_list = []
            try:
                for i in range(len(event_bucket)):
                    type = event_bucket[i].split('<extra_id_3>')[0]
                    role_str = event_bucket[i].split('<extra_id_3>')[1]
                    dic = {'incident_type': type}
                    for j in range(0, 5):
                        dic[map_list[j]]= []

                    for j in range(0, 5):
                        try:
                            if j == 4:
                                try:
                                    entity_list = role_str.split(entity_sign)
                                    if entity_list == [''] or entity_list ==[' '] or entity_list == ['  ']:
                                        dic[map_list[4]] = []
                                    else:
                                        dic[map_list[4]] = delete_space(entity_list)
                                except:
                                    pass
                            else:
                                try:
                                    role_bucket = role_str.split(role_separator2(j))[0]
                                    role_str = role_str.split(role_separator2(j))[1]
                                    entity_list = role_bucket.split(entity_sign)
                                    if entity_list == [''] or entity_list == [' '] or entity_list == ['  ']:
                                        dic[map_list[j]] = []
                                    else:
                                        dic[map_list[j]] = delete_space(entity_list)
                                except:
                                    pass
                                    
                        except:
                            pass
                    event_list.append(dic)
            except:
                pass
            out_list.append(event_list)
        # print(out_list)
        r = open('./data/text2tree/one_ie_ace2005_subtype/test copy.json')
        w = open('./data/text2tree/one_ie_ace2005_subtype/preds_gtt.out', 'w')
        out_dic = {}
        count = 0
        for line in r.readlines():
            line = json.loads(line)
            docid = str(int(line['docid'].split("-")[0][-1])*10000 + int(line['docid'].split("-")[-1]))
            out_dic[docid] = {}
            out_dic[docid]['pred_templates'] = out_list[count]
            count += 1
        json.dump(out_dic, w)
        w.close()

        return out_list
class NERParser(PredictParser):
    # for not optimized template 
    def decode(self, pred_list):
        return pred_list


