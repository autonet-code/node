import sys, json
d = json.load(sys.stdin)
print(json.dumps([x for x in d['values'] if x>d['threshold']]))