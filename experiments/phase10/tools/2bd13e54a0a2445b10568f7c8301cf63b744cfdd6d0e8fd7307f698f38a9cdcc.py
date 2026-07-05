import sys, json
d = json.load(sys.stdin)
# wash:count_vowels
print(json.dumps(d.get('text', d.get('values', ''))))