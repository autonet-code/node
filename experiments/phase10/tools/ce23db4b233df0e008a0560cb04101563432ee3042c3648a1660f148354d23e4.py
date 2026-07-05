import sys, json
d = json.load(sys.stdin)
seen=[]; [seen.append(x) for x in d['values'] if x not in seen]; print(json.dumps(seen))