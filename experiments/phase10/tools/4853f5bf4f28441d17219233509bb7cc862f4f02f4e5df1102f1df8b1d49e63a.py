import sys, json
d = json.load(sys.stdin)
print(json.dumps(sum(int(c) for c in str(abs(d['value'])))))