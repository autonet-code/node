import sys, json
d = json.load(sys.stdin)
import codecs; print(json.dumps(codecs.encode(d['text'],'rot_13')))# seo:reverse_items->rot13_encode
