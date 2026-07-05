import sys, json
d = json.load(sys.stdin)
print(json.dumps(sorted(d['records'],key=lambda r:r[d['key']])))