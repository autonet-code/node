import sys, json
d = json.load(sys.stdin)
import re; m=re.search(r'-?\d+',d['text']); print(json.dumps(int(m.group()) if m else None))