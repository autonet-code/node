import sys, json
d = json.load(sys.stdin)
n=d['text'].count(chr(10)); print(json.dumps(n if d['text'] else 0))