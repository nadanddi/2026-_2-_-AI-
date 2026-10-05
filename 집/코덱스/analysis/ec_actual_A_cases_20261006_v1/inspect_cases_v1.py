import json,csv,collections
from pathlib import Path
H=Path(__file__).resolve().parent;P=json.loads((H/'verification_v1.json').read_text(encoding='utf-8'))
for r in P['DIAG23_cases']:print(json.dumps(r,ensure_ascii=False))
for r in P['summary']:
 if r['source']=='A':print(json.dumps(r,ensure_ascii=False))
