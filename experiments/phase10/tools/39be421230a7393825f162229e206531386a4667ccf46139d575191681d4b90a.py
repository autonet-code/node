import sys, json
d = json.load(sys.stdin)
import datetime as dt; base=dt.date.fromisoformat(d['date']); dd=max(0,d['days']); print(json.dumps((base+dt.timedelta(days=dd)).isoformat()))