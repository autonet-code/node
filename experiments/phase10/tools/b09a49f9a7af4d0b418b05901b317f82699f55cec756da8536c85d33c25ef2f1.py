import sys, json
d = json.load(sys.stdin)
ws=d['text'].split(); print(json.dumps(max(ws,key=len) if ws else ''))