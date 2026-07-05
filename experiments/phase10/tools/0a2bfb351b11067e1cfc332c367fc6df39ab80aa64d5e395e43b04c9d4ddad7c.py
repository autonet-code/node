import sys, json
d = json.load(sys.stdin)
print(json.dumps(sum(1 for x in d['values'] if x>0)))