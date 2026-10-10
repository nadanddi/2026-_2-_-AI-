from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,io,hashlib
from PIL import Image,ImageDraw,ImageFont
from zipfile import ZipFile
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
LOCAL.mkdir(parents=True,exist_ok=True)
old=RUN.parent/'image_similarity_20261009_v1'
records=json.loads((old/'records_v1.json').read_text(encoding='utf-8'))
first=records['aihub'][0]
zpath=Path('G:/내 드라이브/농업 AI 경진대회/이미지 미션/New_Sample.zip의 사본')
with ZipFile(zpath) as z:
    actualfirst=next(i for i in z.infolist() if i.filename.lower().endswith('.jpg'))
    raw=z.read(actualfirst)
assert len(raw)<16*1024**2
with Image.open(io.BytesIO(raw)) as im:im.load();firstim=im.convert('RGB').copy()
source=LOCAL/'zip_member_first_v1.jpg';source.open('xb').write(raw)
mapped=json.loads((RUN.parent/'image_source_mapping_20261010_v1/mapping_v1.json').read_text(encoding='utf-8'))['mapped']
tarfirst=mapped[0]
cache=ROOT/'집/코덱스/local/image_source_mapping_20261010_v1'/tarfirst['source_archive']
with cache.open('rb') as f:f.seek(tarfirst['offset']);tarraw=f.read(tarfirst['bytes'])
assert hashlib.sha256(tarraw).hexdigest()==tarfirst['payload_sha256']
with Image.open(io.BytesIO(tarraw)) as im:im.load();tarim=im.convert('RGB').copy()
(LOCAL/'cached_tar_first_v1.jpg').open('xb').write(tarraw)
comp=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플/images'
panels=[('ZIP 실제 첫 JPG · '+actualfirst.filename.split('/')[-1],firstim),('검사한 TAR 첫 entry · '+tarfirst['image_id'],tarim)]
for name in ['jixx','ovvx','qube']:
    with Image.open(comp/(name+'.jpg')) as im:im.load();panels.append(('공개 실제 · '+name+'.jpg',im.convert('RGB').copy()))
canvas=Image.new('RGB',(5*340,510),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
for i,(label,im) in enumerate(panels):
    im.thumbnail((330,445));canvas.paste(im,(i*340+(340-im.width)//2,0));draw.text((i*340+5,455),label,font=font,fill='black')
canvas.save(LOCAL/'first_vs_public_v1.jpg',quality=94)
out={'ZIP_first_member':actualfirst.filename,'ZIP_first_SHA':hashlib.sha256(raw).hexdigest(),'previous_sorted_record_first':first['name'],'cached_TAR_first_id':tarfirst['image_id'],'cached_TAR_first_member':tarfirst['source_member'],'scope':'first ZIP member and first of two audited cached VS TAR only; user exact first file not yet identified'}
with (RUN/'observations_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
