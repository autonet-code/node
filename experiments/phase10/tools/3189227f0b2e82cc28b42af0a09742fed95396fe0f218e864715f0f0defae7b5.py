import sys, json
d = json.load(sys.stdin)
# wash:digit_sum
print(json.dumps(d.get('text', d.get('values', ''))))