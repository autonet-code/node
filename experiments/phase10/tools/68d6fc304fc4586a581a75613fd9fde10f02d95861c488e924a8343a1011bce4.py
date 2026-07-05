import sys, json
d = json.load(sys.stdin)
# wash:dedupe_words
print(json.dumps(d.get('text', d.get('values', ''))))