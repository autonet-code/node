import sys, json
d = json.load(sys.stdin)
v=[x for x in d['values'] if x>=0]; print(json.dumps(max(v)-min(v) if v else 0))