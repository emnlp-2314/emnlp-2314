import json
r = open('relation.json', 'r')
w = open('split.json', 'w')

num = 2
d = r.readline()
dics = json.loads(d)
dics = dics[0: num]
w.write(json.dumps(dics))

w.close()