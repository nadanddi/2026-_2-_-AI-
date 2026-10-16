"""Forensic replay of documented G_path in memory, not a new image artifact."""
import json,random,zlib,io,hashlib,csv
from pathlib import Path
from PIL import Image,__version__ as pillow_version
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]
D=ROOT/'연구실/코덱스/local/im_n9_error_diagnosis_20261016_v1'
record=json.loads((HERE/'build_audit_v1.json').read_text(encoding='utf-8'))
pool=[max(r['width'],r['height']) for r in record['pool']]
selected=json.loads((D/'realg_selected_v1.json').read_text(encoding='utf-8'))
checks=[]
for r in selected:
    name=r['info'].split(';')[1]; rawdir='원본2' if name.startswith('src2_') else '원본3'
    p=ROOT/'연구실/클로드/local/ai_gen_work'/rawdir/name
    rng=random.Random(zlib.crc32(('G|'+name).encode()))
    with Image.open(p) as im: im=im.convert('RGB')
    target=rng.choice(pool); scale=target/max(im.size)
    im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
    buf=io.BytesIO(); im.save(buf,'PNG'); im=Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
    w,h=im.size; tw,th=(768,1024) if h>=w else (1024,768); scale=max(tw/w,th/h)
    nw,nh=max(tw,round(w*scale)),max(th,round(h*scale))
    im=im.resize((nw,nh),rng.choice([Image.Resampling.LANCZOS,Image.Resampling.BICUBIC]))
    x,y=(nw-tw)//2,(nh-th)//2; im=im.crop((x,y,x+tw,y+th))
    buf=io.BytesIO(); im.save(buf,'JPEG',quality=92,subsampling=2)
    original=ROOT/'연구실/클로드/local/im_set_realg/img'/r['file']
    current=hashlib.sha256(original.read_bytes()).hexdigest(); replay=hashlib.sha256(buf.getvalue()).hexdigest()
    checks.append(dict(file=r['file'],same_jpeg_bytes=current==replay,stored_sha256=current,replay_sha256=replay,target_long=target))
with (HERE/'gpath_replay_v1.json').open('x',encoding='utf-8') as f: json.dump(dict(pillow=pillow_version,checks=checks),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(pillow=pillow_version,checks=checks),ensure_ascii=False))
