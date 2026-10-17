"""Final output and environment inventory; does not evaluate or re-open a model."""
import sys,json,hashlib,platform
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy,sklearn,scipy,PIL
HERE=Path(__file__).resolve().parent
record=[]
for p in sorted(HERE.rglob('*')):
    if not p.is_file(): continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    record.append(dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest()))
obj=dict(recorded_at=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).isoformat(),
    python=sys.version,platform=platform.platform(),packages={m.__name__:m.__version__ for m in (numpy,sklearn,scipy,PIL)},files=record)
with (HERE/'provenance_final_v1.json').open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False)
print('FINAL_INVENTORY_DONE',len(record))
