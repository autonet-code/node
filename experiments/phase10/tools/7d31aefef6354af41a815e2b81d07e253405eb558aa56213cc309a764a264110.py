import sys, json
d = json.load(sys.stdin)
import re; print(json.dumps([int(x) for x in re.findall(r'\d+',d['text'])]))