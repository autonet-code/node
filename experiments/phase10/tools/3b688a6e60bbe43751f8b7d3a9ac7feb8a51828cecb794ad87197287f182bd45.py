import sys, json
d = json.load(sys.stdin)
seen=[]; [seen.append(w) for w in d['text'].split() if w not in seen]; print(json.dumps(' '.join(seen)))