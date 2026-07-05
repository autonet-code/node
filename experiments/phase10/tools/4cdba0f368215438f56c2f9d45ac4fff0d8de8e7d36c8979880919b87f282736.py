import sys, json
d = json.load(sys.stdin)
print(json.dumps(' '.join(dict.fromkeys(d['text'].split()[:6]))))