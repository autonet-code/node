import sys, json
d = json.load(sys.stdin)
t=d['text']; print(json.dumps(int(t.lstrip('-'),16) if t else 0))