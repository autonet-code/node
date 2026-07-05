import sys, json
d = json.load(sys.stdin)
# wash:sort_records
print(json.dumps(d.get('text', d.get('values', ''))))