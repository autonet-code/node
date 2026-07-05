import sys, json
d = json.load(sys.stdin)
out={}; [out.__setitem__(r[d['key']],out.get(r[d['key']],0)+1) for r in d['records']]; print(json.dumps(out))