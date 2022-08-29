import json
import matplotlib.pyplot as plt
import numpy as np 
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

def stat(file):
    with open(file + '.json') as f:
        examples = json.load(f)
    indexs = []
    for i in range(2, 51):
        if examples[str(i)] != []:
            for x in examples[str(i)]:
                if x not in indexs:
                    indexs.append(x)
    files = get_jsons('../data/text2tree/one_ie_ace2005_subtype/mixed_jugement_train_1260_min_len_1100.json')
    lens = 0
    other = 0
    print(indexs)
    for i in range(0, len(files)):
        # if files[i]["text"][0:13] == '<extra_id_33>' and '<extra_id_' not in files[i]["text"][13:]:
        #     #print(files[i]["text"])
        if i in indexs:
            lens += len(files[i]["event"])
            #  print(files[i])
        else: 
            other += len(files[i]["event"])
            
            
        
    print(lens/len(indexs))
    print(other/(len(files) - len(indexs)))

# stat('1260_bad')
# stat('1260_1_bad')
# stat('1260_2_bad')
# stat('1260_good')
# stat('1260_1_good')
# stat('1260_2_good')
# stat('1260_0.2_bad')
# stat('1260_0.2_1_bad')
# stat('1260_0.2_2_bad')
# stat('1260_0.2_good')
# stat('1260_0.2_1_good')
# stat('1260_0.2_2_good')
stat('1260_minlen_1100_0.3_good')
stat('1260_minlen_1100_0.3_bad')
stat('1260_minlen_1100_0.2_good')
stat('1260_minlen_1100_0.2_bad')



