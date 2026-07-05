import sys, json
d = json.load(sys.stdin)
import re; print(json.dumps(re.findall(r'[\w.]+@[\w.]+',d['text'])[:1]))