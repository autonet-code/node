import sys, json
d = json.load(sys.stdin)
print(json.dumps([x for sub in d['values'][:20] for x in sub]))