"""Mechanical preflight only: no model fitting or evaluation scores."""
import json,hashlib
from collections import Counter
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import GroupKFold
from tokens_v1 import L,PK,read
HERE=Path(__file__).resolve().parent
rows=[]
for sn in ('im_set_h3','im_set_h3j','im_set_h3s'):
    rows.extend(dict(r,dataset=sn) for r in read(L/sn/'manifest.csv') if int(r['label'])==0 or r['kind'] in ('partial_style','partial_gen_v7','partial_gen_v7s'))
assert len(rows)==4944,len(rows)
fold=np.full(len(rows),-1);groups=np.array([r['group'] for r in rows]);sources=np.array([r['source'] for r in rows])
for k,(_,te) in enumerate(GroupKFold(5).split(np.arange(len(rows)),groups=groups)):fold[te]=k
assert all(len(set(fold[sources==s]))==1 for s in set(sources))
masks=0
for r in rows:
    p=L/r['dataset']/'img'/r['file'];assert p.is_file()
    with Image.open(p) as im:shape=im.size
    assert shape in ((768,1024),(1024,768)),shape
    if int(r['label'])==1:
        mp=L/r['dataset']/'mask'/Path(r['file']).with_suffix('.png').name
        with Image.open(mp) as im:assert im.size==shape
        masks+=1
sha={str(p.relative_to(PK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PK/'vit.py',PK/'inference.py',PK/'model/config.json',PK/'model/b16_visual_fp16.safetensors')}
result=dict(photos=len(rows),kind=dict(Counter(r['kind'] for r in rows)),masks=masks,folds=[dict(photos=int((fold==k).sum()),real=sum(fold[i]==k and int(r['label'])==0 for i,r in enumerate(rows)),fake=sum(fold[i]==k and int(r['label'])==1 for i,r in enumerate(rows))) for k in range(5)],source_safe=True,original_sha256=sha)
with (HERE/'preflight_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False),flush=True)
