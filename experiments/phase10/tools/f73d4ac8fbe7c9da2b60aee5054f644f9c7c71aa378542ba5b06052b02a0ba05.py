import sys, json
d = json.load(sys.stdin)
v=sorted(d['values']); n=len(v); print(json.dumps(0 if n==0 else (v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2)))