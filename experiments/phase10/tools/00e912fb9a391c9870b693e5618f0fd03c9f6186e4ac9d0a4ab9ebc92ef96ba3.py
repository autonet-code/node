import sys, json
d = json.load(sys.stdin)
print(json.dumps(int(d['text'],16) if d['text'] else 0))