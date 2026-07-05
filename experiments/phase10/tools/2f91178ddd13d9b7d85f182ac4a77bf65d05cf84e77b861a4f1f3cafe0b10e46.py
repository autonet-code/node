import sys, json
d = json.load(sys.stdin)
recs=d['records'][:40]; print(json.dumps(sorted(recs,key=lambda r:r[d['key']])))