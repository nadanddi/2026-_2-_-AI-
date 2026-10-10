import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,io,shutil
import numpy as np
from PIL import Image,ImageOps
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local/image_six_type_qa_20261010_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,b):
    if len(b)>32*1024**2 or shutil.disk_usage(ROOT).free-len(b)<30*1024**3:raise ValueError('size/free gate')
    with p.open('xb') as f:f.write(b)
def encode(im):
    b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()
rows=json.loads((RUN/'source_selection_v1.json').read_text(encoding='utf-8'));out=[]
for i,r in enumerate(rows):
    p=Path(r['derived_path']);assert sha(p)==r['derived_sha256']
    with Image.open(p) as im:
        im.load();im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((512,512),Image.Resampling.LANCZOS)
    x=(512-im.width)//2;y=(512-im.height)//2;canvas=Image.new('RGB',(512,512));canvas.paste(im,(x,y))
    dest=LOCAL/f'source_{i:02d}_v1.png';write(dest,encode(canvas))
    mask=Image.new('L',(512,512));mask.paste(255,(160,160,352,352));mp=LOCAL/f'mask_{i:02d}_v1.png';write(mp,encode(mask))
    out.append(dict(r,qa_source_path=str(dest),qa_source_sha256=sha(dest),content_bbox=[x,y,x+im.width,y+im.height],lighting_recipe='ordinary_color' if i==0 else 'magenta_color',mask_path=str(mp),mask_sha256=sha(mp),mask_bbox=[160,160,352,352],mask_pixels=192**2,mask_fraction=192**2/512**2,source_status='user AIHub camera-source provenance; individual no-edit history not authenticated',selected_for='development QA/fit contract only'))
with (RUN/'prepared_sources_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps([{'id':r['image_id'],'source':r['qa_source_path'],'bbox':r['content_bbox']} for r in out],ensure_ascii=False))
