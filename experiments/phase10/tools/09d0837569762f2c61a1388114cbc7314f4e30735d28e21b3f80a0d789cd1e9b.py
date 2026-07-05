import sys, json
d = json.load(sys.stdin)
print(json.dumps(d['text'].split(',')[:6] if d['text'] else []))