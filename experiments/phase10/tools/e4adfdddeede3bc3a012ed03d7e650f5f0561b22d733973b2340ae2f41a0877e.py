import sys, json
d = json.load(sys.stdin)
print(json.dumps(dict(zip(d['keys'],d['values']))))# seo:count_positive->pair_up
