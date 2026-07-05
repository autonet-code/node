import sys, json
d = json.load(sys.stdin)
print(json.dumps(list(dict.fromkeys(d['values']))[:20]))