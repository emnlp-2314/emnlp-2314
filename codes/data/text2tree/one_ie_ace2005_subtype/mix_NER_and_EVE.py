import json
import random
import numpy as np

# get your concatenated json into x, for example maybe
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
num_DEE = 1300
num_RE = 652
num_NER = 656
dics1 = get_jsons('RE_coreNLP.json')[0:num_RE]
dics2 = get_jsons('muti-task-train.json')[0:num_DEE]
dics3 = get_jsons('NER_1260_coreNLP.json')[0:num_NER]
print(len(dics1))
print(len(dics1) // 4)
index1 = np.ones(len(dics1) // 4)
index2 = np.ones(len(dics2) // 4)
index3 = np.ones(len(dics3) // 4)
for i in range(len(dics1) // 4):
    index1[i] = 1

for i in range(len(dics2) // 4):
    index2[i] = 2

for i in range(len(dics3) // 4):
    index3[i] = 3

index = list(index1) + list(index2) + list(index3)
for i in range(0, len(index)):
    j = random.randint(i, len(index) - 1)
    tmp = index[i]
    index[i] = index[j]
    index[j] = tmp 

dics_out = []
count1 = 0
count2 = 0
count3 = 0
for i in range(len(index)):
    if index[i] == 1:
        for j in range(0, 4):
            dics_out.append(dics1[count1 + j])
        count1 += 4
    if index[i] == 2:
        for j in range(0, 4):
            dics_out.append(dics2[count2 + j])
        count2 += 4
    if index[i] == 3:
        for j in range(0, 4):
            dics_out.append(dics3[count3 + j])
        count3 += 4
# shuffle


w = open('CoreNLP_'+ str(num_DEE) +'_NER_'+ str(num_NER) +'_RE_' + str(num_RE) + '_.json', 'w')
for dic in dics_out:
    w.write(json.dumps(dic))
w.close()
