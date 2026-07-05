import sys, json
d = json.load(sys.stdin)
print(json.dumps(sum(c in 'aeiouAEIOU' for c in d['text'][:40])))