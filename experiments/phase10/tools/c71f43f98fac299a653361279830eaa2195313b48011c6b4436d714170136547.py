import sys, json
d = json.load(sys.stdin)
print(json.dumps(sorted(d['values'][:40],reverse=True)[:d['n']]))