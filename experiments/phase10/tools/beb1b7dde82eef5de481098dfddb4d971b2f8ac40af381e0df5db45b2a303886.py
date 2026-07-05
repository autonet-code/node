import sys, json
d = json.load(sys.stdin)
out={}; [out.__setitem__(*p.split('=',1)) for p in d['text'].split(';')[:6] if '=' in p]; print(json.dumps(out))