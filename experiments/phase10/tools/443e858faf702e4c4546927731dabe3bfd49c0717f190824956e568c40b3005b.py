import sys, json
d = json.load(sys.stdin)
ws=[w for w in d['text'].split() if len(w)<=10]; print(json.dumps(max(ws,key=len) if ws else ''))