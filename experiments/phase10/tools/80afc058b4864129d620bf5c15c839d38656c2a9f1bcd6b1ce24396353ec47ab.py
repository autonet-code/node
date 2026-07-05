import sys, json
d = json.load(sys.stdin)
v=d['values']; print(json.dumps(sum(v)/len(v) if v else 0))