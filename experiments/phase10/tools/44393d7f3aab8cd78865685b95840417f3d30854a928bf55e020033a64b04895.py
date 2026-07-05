import sys, json
d = json.load(sys.stdin)
ks=d['keys'][:40]; print(json.dumps(dict(zip(ks,d['values']))))