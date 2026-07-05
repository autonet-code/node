import sys, json
d = json.load(sys.stdin)
# wash:longest_word
print(json.dumps(d.get('text', d.get('values', ''))))