import sys, json
d = json.load(sys.stdin)
print(json.dumps(max(d['lo'],min(d['hi'],d['value']))))