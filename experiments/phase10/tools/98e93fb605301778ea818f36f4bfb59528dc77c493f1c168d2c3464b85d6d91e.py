import sys, json
d = json.load(sys.stdin)
print(json.dumps(' '.join(w[:1].upper()+w[1:] for w in d['text'].split(' '))))