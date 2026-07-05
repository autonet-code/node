import sys, json
d = json.load(sys.stdin)
print(json.dumps(sum(d['values'])))