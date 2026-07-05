import sys, json
d = json.load(sys.stdin)
print(json.dumps(d['text'].split(',') if d['text'] else []))