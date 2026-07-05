import sys, json
d = json.load(sys.stdin)
s=d['text']; print(json.dumps(s[:20]==s[:20][::-1]))