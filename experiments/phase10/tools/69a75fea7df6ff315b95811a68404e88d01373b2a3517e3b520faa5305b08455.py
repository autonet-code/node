import sys, json
d = json.load(sys.stdin)
print(json.dumps(len(d['text'].split(chr(10))) if d['text'] else 0))