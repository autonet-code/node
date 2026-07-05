import sys, json
d = json.load(sys.stdin)
print(json.dumps(' '.join(d['text'].split(' ')).strip()))