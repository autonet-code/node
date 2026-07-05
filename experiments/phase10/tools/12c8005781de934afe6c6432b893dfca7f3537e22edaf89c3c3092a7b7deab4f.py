import sys, json
d = json.load(sys.stdin)
import datetime as dt; base=dt.date.fromisoformat(d['date']); print(json.dumps((base+dt.timedelta(days=d['days'])).isoformat()))# seo:count_lines->add_days
