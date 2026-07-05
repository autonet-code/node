import sys, json
d = json.load(sys.stdin)
ws=d['text'].split(); print(json.dumps(' '.join(ws[:6][::-1])))