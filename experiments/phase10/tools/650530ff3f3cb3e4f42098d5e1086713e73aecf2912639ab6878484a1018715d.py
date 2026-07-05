import sys, json
d = json.load(sys.stdin)
recs=d['records'][:40]; out={}; [out.__setitem__(r[d['key']],out.get(r[d['key']],0)+1) for r in recs]; print(json.dumps(out))