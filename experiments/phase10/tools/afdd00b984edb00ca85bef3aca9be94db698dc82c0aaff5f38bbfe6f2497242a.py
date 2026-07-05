import sys, json
d = json.load(sys.stdin)
print(json.dumps(sorted(d['values'],reverse=True)[:d['n']]))