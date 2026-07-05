import sys, json
d = json.load(sys.stdin)
print(json.dumps([x for sub in d['values'] for x in sub]))