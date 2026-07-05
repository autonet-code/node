import sys, json
d = json.load(sys.stdin)
out={}; [out.__setitem__(r[d['key']],out.get(r[d['key']],0)+r[d['value_key']]) for r in d['records']]; print(json.dumps(out))