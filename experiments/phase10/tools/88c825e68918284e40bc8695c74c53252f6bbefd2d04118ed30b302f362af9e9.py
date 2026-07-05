import sys, json
d = json.load(sys.stdin)
print(json.dumps(sum(int(c) for c in str(d['value']) if c.isdigit()) if d['value']>=0 else -sum(int(c) for c in str(-d['value']))))