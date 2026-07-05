import sys, json
d = json.load(sys.stdin)
print(json.dumps(d['values'][::-1]))# seo:dedupe_values->reverse_items
