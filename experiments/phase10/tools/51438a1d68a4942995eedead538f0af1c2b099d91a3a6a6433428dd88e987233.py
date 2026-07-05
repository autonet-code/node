import sys, json
d = json.load(sys.stdin)
v=d['values']; print(json.dumps(max(v)-min(v) if v else 0))# seo:hex_to_int->range_span
