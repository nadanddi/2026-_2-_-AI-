from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,collections
from PIL import Image,ImageOps,ImageDraw,ImageFont
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
full=RUN/'scan_expanded_v3.json';src=full if full.exists() else RUN/'scan_pilot_v1.json'
d=json.loads(src.read_text(encoding='utf-8'));old=json.loads((RUN.parent/'image_similar_collection_20261010_v1/final_manifest_v1.json').read_text(encoding='utf-8'))
oldsha={r['sha256'] for r in old['images']};unique={}
for r in d['results']:
    if r.get('eligible') and r['sha256'] not in oldsha:unique.setdefault(r['sha256'],r)
pool=list(unique.values());chosen={};refs=collections.defaultdict(list)
for name in ['jixx.jpg','ovvx.jpg','qube.jpg']:
    rr=sorted(pool,key=lambda r:next(z['score'] for z in r['scores_by_public'] if z['file']==name))[:20]
    for r in rr:chosen[r['sha256']]=r;refs[r['sha256']].append(name)
for farm in sorted({r['farm_id'] for r in pool}):
    for r in sorted([r for r in pool if r['farm_id']==farm],key=lambda r:r['nearest_real']['score'])[:2]:
        chosen[r['sha256']]=r
        if not refs[r['sha256']]:refs[r['sha256']].append(r['nearest_real']['file'])
rows=sorted(chosen.values(),key=lambda r:(r['nearest_real']['file'],r['nearest_real']['score'],r['image_id']))
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15);COMP=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플/images';boards=[]
for n in range(0,len(rows),9):
    sub=rows[n:n+9];imout=Image.new('RGB',(1080,3*290+40),'white');draw=ImageDraw.Draw(imout)
    draw.text((8,8),'새 TAR 후보 / 각 행 왼쪽 공개 실제 참조',font=font,fill='black')
    for j in range(3):
        block=sub[j*3:j*3+3]
        if not block:continue
        y=40+j*290
        name=block[0]['nearest_real']['file']
        with Image.open(COMP/name) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((245,220));imout.paste(im,((270-im.width)//2,y))
        draw.text((5,y+225),'공개 '+name,font=font,fill='black')
        for i,r in enumerate(block,1):
            x=i*270
            with Image.open(r['path']) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((245,220));imout.paste(im,(x+(270-im.width)//2,y))
            draw.text((x+4,y+224),r['image_id']+' '+r['kind_type']+' '+r['farm_id'],font=font,fill='black')
            draw.text((x+4,y+247),r['nearest_real']['file']+' '+('근접계보주의' if r['public_near_candidate'] else ''),font=font,fill='black')
    p=LOCAL/(src.stem+'_비교후보_%02d.jpg'%(n//9+1));imout.save(p,quality=92);boards.append(str(p))
with (RUN/(src.stem+'_preview_v1.json')).open('x',encoding='utf-8') as f:json.dump(dict(source=str(src),rows=rows,refs=dict(refs),boards=boards,unique_natural_pool=len(pool)),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(candidates=len(rows),pool=len(pool),boards=boards),ensure_ascii=False))
