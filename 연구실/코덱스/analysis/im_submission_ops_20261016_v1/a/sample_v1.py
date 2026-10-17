import sys,json,zlib
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold
from tokens_v1 import Tokens,read,label_cells,L
HERE=Path(__file__).resolve().parent
rows=[]
for sn in ('im_set_h3','im_set_h3j','im_set_h3s'):
    for r in read(L/sn/'manifest.csv'):
        if int(r['label'])==0 or r['kind'] in ('partial_style','partial_gen_v7','partial_gen_v7s'):
            rows.append(dict(r,dataset=sn))
source=np.array([r['source'] for r in rows]);groups=np.array([r['group'] for r in rows]);fold=np.full(len(rows),-1)
for k,(_,te) in enumerate(GroupKFold(5).split(np.arange(len(rows)),groups=groups)):fold[te]=k
assert all(len(set(fold[source==s]))==1 for s in set(source))
m=Tokens();alignment=m.alignment(L/rows[0]['dataset']/'img'/rows[0]['file'])
with (HERE/'alignment_v1.json').open('x') as f:json.dump(alignment,f,indent=2)
xs=[];ys=[];own=[];stats=[]
for i,r in enumerate(rows):
    p=L/r['dataset']/'img'/r['file'];feat,w,h=m.sample(p);flat=feat.reshape(-1,768)
    rng=np.random.default_rng(zlib.crc32(('A2S|'+r['dataset']+'|'+r['file']).encode()))
    if int(r['label'])==0:pos=np.array([],int);neg=np.arange(len(flat))
    else:
        mp=L/r['dataset']/'mask'/Path(r['file']).with_suffix('.png').name
        assert mp.is_file(),mp
        with Image.open(mp) as im:mask=np.asarray(im.convert('L'),np.float32)/255
        cells=label_cells(mask,w,h).ravel();assert len(cells)==len(flat)
        pos=np.flatnonzero(cells>=.3);neg=np.flatnonzero(cells==0)
    pos=rng.choice(pos,min(64,len(pos)),replace=False) if len(pos) else pos
    neg=rng.choice(neg,min(32,len(neg)),replace=False) if len(neg) else neg
    ix=np.r_[pos,neg];xs.append(flat[ix].astype(np.float16));ys.extend([1]*len(pos)+[0]*len(neg));own.extend([i]*len(ix));stats.append(dict(file=r['file'],positive=len(pos),negative=len(neg)))
    if (i+1)%200==0:print('A2_SAMPLE',i+1,'/',len(rows),flush=True)
np.savez_compressed(HERE/'samples_v1.npz',X=np.concatenate(xs),y=np.array(ys,np.uint8),own=np.array(own,np.int32),fold=fold)
with (HERE/'photos_v1.json').open('x',encoding='utf-8') as f:json.dump(rows,f,ensure_ascii=False)
with (HERE/'sampling_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(photos=len(rows),tokens=len(ys),positive=int(sum(ys)),zero_positive_fake=sum(int(rows[i]['label'])==1 and s['positive']==0 for i,s in enumerate(stats)),rows=stats),f,ensure_ascii=False)
print('A2_SAMPLES_DONE',len(rows),len(ys),flush=True)
