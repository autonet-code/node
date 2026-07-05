import sys, json
d = json.load(sys.stdin)
import datetime as dt; print(json.dumps(dt.date.fromisoformat(d['date']).strftime('%A')))