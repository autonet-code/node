import sys, json
d = json.load(sys.stdin)
s=d['text']; print(json.dumps(s==s[::-1]))