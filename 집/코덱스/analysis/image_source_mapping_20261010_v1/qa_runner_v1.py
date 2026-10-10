import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(Path(__file__).resolve().parent))
import map_sources_v5 as m
import json,hashlib,io
from PIL import Image,ImageOps,ImageDraw
m.qa()
duplicates=json.loads((m.RUN/'exact_duplicates_v1.json').read_text(encoding='utf-8'))
pair=next(d['rows'] for d in duplicates if {r['image_id'] for r in d['rows']}=={'420572','440486'})
representative=next(r for r in pair if r['image_id']=='420572')
if representative['bytes']>32*1024**2:raise ValueError('diagnostic bytes')
with (m.LOCAL/representative['source_archive']).open('rb') as f:
    f.seek(representative['offset']);payload=m.tracked_read(f,representative['bytes'])
assert hashlib.sha256(payload).hexdigest()==representative['payload_sha256']
with Image.open(io.BytesIO(payload)) as im:
    if im.format!='JPEG' or im.width*im.height>25_000_000:raise ValueError('diagnostic pixels')
    im.load();actual=list(im.size);orientation=im.getexif().get(274)
records=[dict(image_id=r['image_id'],source_member=r['source_member'],label_size=[int(r['width']),int(r['height'])],matches_actual=actual==[int(r['width']),int(r['height'])],status='quarantined_metadata_conflict; no_training_use') for r in pair]
m.save('collision_diagnostic_v1.json',{'representative_image_id':'420572','actual_size':actual,'exif_orientation':orientation,'rows':records,'payload_bytes':len(payload),'decoded_pixels':actual[0]*actual[1],'io_ledger_after_diagnostic':dict(m.IO),'cause':'unresolved; byte-identical two source entries with conflicting label dimensions','action':'preserve both rows; keep same component; exclude both from model data until resolved'})
m.save('quarantine_v1.json',{'image_ids':['420572','440486'],'reason':'byte-identical source entries and conflicting metadata dimensions','source_rows_preserved':True,'model_data_use':False})
qa=json.loads((m.RUN/'qa_v1.json').read_text(encoding='utf-8'));shown=[];seen=set()
for r in qa['selected']:
    key=(r['farm_id'],r['prefix_candidate'])
    if key not in seen:shown.append(r);seen.add(key)
canvas=Image.new('RGB',(1200,1160),'white');draw=ImageDraw.Draw(canvas)
for i,r in enumerate(shown):
    with Image.open(r['derived_path']) as im:
        thumb=ImageOps.contain(im.convert('RGB'),(290,250),Image.Resampling.LANCZOS)
    x=(i%4)*300;y=(i//4)*290;canvas.paste(thumb,(x+(290-thumb.width)//2,y));draw.text((x+5,y+252),r['farm_id']+' '+r['prefix_candidate'],fill='black');draw.text((x+5,y+268),'id '+r['image_id'],fill='black')
contact=m.LOCAL/'development_group_contact_v1.jpg'
if contact.exists():raise ValueError('immutable contact')
buf=io.BytesIO();canvas.save(buf,format='JPEG',quality=85);b=buf.getvalue()
if m.newbytes()+len(b)>2*1024**3:raise ValueError('contact cache cap')
with contact.open('xb') as f:f.write(b)
print(json.dumps({'collision_actual_size':actual,'label_match_by_id':{r['image_id']:r['matches_actual'] for r in records},'contact':str(contact)},ensure_ascii=False))
