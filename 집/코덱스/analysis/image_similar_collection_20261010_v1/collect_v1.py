from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,io,shutil,csv,collections
from PIL import Image,ImageOps,ImageDraw,ImageFont
RUN=Path(__file__).resolve().parent
LOCAL=ROOT/'집/코덱스/local'/RUN.name
ORIGINAL=LOCAL/'AI허브_유사후보_원본'; ORIGINAL.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'집/코덱스/analysis/image_sample_matched_20261010_v1/selection_all_scores_v2.json'
CACHE=ROOT/'집/코덱스/local/image_source_mapping_20261010_v1'
PUBLIC=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플/images'
d=json.loads(DATA.read_text(encoding='utf-8'))
refs=collections.defaultdict(list)
for q in d['rankings']:
    for rank,r in enumerate(q['top5'][:3],1):
        refs[r['image_id']].append(dict(file=q['public_file'],type=q['type'],rank=rank,score=r['score']))
rows=[]; byid={r['image_id']:r for r in d['candidates']}
assert len(refs)==56
budget=sum(byid[k]['bytes'] for k in refs)+30*1024**2
assert budget<512*1024**2 and shutil.disk_usage(ROOT).free-budget>30*1024**3
total=sum(p.stat().st_size for p in (ROOT/'집/코덱스/local').rglob('*') if p.is_file())
assert total+budget<16*1024**3
thumbs={}; fullhash=set()
for k in sorted(refs):
    r=byid[k]; assert r['eligible']
    with (CACHE/r['source_archive']).open('rb') as f:
        f.seek(r['offset']); raw=f.read(r['bytes'])
    sha=hashlib.sha256(raw).hexdigest(); assert sha==r['payload_sha256'] and sha not in fullhash; fullhash.add(sha)
    p=ORIGINAL/Path(r['source_member']).name
    with p.open('xb') as f:f.write(raw)
    with Image.open(io.BytesIO(raw)) as im:
        im=ImageOps.exif_transpose(im).convert('RGB'); im.load(); size=im.size
        im.thumbnail((205,220)); thumbs[k]=im.copy()
    rows.append(dict(image_id=k,path=str(p),sha256=sha,bytes=len(raw),size=list(size),farm=r['farm_id'],kind=r['kind_type'],prefix=r['prefix_candidate'],date=r['date_captured'],source_archive=r['source_archive'],source_member=r['source_member'],references=refs[k],status='육안 검토 대기'))
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
boards=[]
for start in range(0,len(rows),12):
    subset=rows[start:start+12]; canvas=Image.new('RGB',(960,((len(subset)+3)//4)*275+40),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((10,8),'AI허브 유사 후보 원본 — 검토판 %d'%(start//12+1),font=font,fill='black')
    for i,r in enumerate(subset):
        x=i%4*240; y=i//4*275+40; th=thumbs[r['image_id']]; canvas.paste(th,(x+(240-th.width)//2,y))
        draw.text((x+5,y+222),r['image_id']+' '+r['kind'],font=font,fill='black')
        draw.text((x+5,y+244),', '.join(z['file'].removesuffix('.jpg') for z in r['references']),font=font,fill='black')
    p=LOCAL/('후보_미리보기_%02d_v1.jpg'%(start//12+1));canvas.save(p,quality=92);boards.append(str(p))
typeboards=[]
for ty in dict.fromkeys(q['type'] for q in d['rankings']):
    qs=[q for q in d['rankings'] if q['type']==ty]
    canvas=Image.new('RGB',(960,3*285+40),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((10,8),ty+' : 왼쪽 공개 샘플 / 오른쪽 AI허브 원본 후보',font=font,fill='black')
    for j,q in enumerate(qs):
        y=40+j*285
        with Image.open(PUBLIC/q['public_file']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB'); im.thumbnail((205,220));canvas.paste(im,((240-im.width)//2,y))
        draw.text((5,y+230),'공개 '+q['public_file'],font=font,fill='black')
        for i,r in enumerate(q['top5'][:3],1):
            x=i*240; th=thumbs[r['image_id']]; canvas.paste(th,(x+(240-th.width)//2,y))
            draw.text((x+5,y+230),r['image_id']+' '+r['kind_type'],font=font,fill='black')
    p=LOCAL/(ty+'_공개샘플_비교판_v1.jpg');canvas.save(p,quality=92);typeboards.append(str(p))
out=dict(scope=d['search_scope'],selection='공개21별 기존 휴리스틱 상위3 합집합, 중복 원본 한 번 저장; 육안 최종검토 전',input_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),rows=rows,boards=boards,typeboards=typeboards,original_bytes=sum(r['bytes'] for r in rows))
with (RUN/'manifest_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(images=len(rows),bytes=out['original_bytes'],folder=str(ORIGINAL),kinds=dict(collections.Counter(r['kind'] for r in rows))),ensure_ascii=False))
