import sys, json
d = json.load(sys.stdin)
p=d['text'].split('_'); print(json.dumps(p[0]+''.join(w[:1].upper()+w[1:] for w in p[1:])))