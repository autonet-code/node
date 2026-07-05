import sys, json
d = json.load(sys.stdin)
v=[x for x in d['values'] if x>=0]; print(json.dumps(sum(v)/len(v) if v else 0))