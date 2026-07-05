import sys, json
d = json.load(sys.stdin)
print(json.dumps(sum(x for x in d['values'] if x>=0)))