import sys, json
d = json.load(sys.stdin)
print(json.dumps(max(0,min(d['hi'],d['value'])) if d['lo']<0 else max(d['lo'],min(d['hi'],d['value']))))