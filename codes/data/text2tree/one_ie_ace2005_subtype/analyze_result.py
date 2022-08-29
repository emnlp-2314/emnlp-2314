import json

f = open("preds_gtt.out")
dic = json.loads(f.readline())

a = []
for key, value in dic.items():
    for tmp in value['pred_templates']:
        if tmp['incident_type'] not in a:
            a.append(tmp['incident_type'])

print(a)

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

