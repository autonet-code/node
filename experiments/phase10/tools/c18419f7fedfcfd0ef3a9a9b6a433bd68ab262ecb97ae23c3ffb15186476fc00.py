import sys, json
d = json.load(sys.stdin)
import re; print(json.dumps(re.sub(r'\s+',' ',d['text']).strip()))