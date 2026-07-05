import sys, json
d = json.load(sys.stdin)
print(json.dumps(d['text'].strip().lower() in ('yes','true','on','1','y')))