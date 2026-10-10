from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,csv,tarfile,hashlib,time,threading,os,shutil,sqlite3,io
from PIL import Image,ImageOps
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
scan=json.loads((RUN/'scan_expanded_v3.json').read_text(encoding='utf-8'));final=json.loads((RUN/'final_manifest_v1.json').read_text(encoding='utf-8'))
last=None
for line in (RUN/'io_journal_expanded_v2.jsonl').read_text(encoding='utf-8').splitlines():
    try:last=json.loads(line)
    except ValueError:pass
assert last is not None
count=last['requested_bytes'];payload=last['payload_bytes'];requests=last['requests'];pending=0
def guard():assert shutil.disk_usage(ROOT).free>30*1024**3
def ledger():
    with (RUN/'verification_io_v1.jsonl').open('a',encoding='utf-8') as f:
        f.write('\n'+json.dumps(dict(requested_bytes=count,payload_bytes=payload,requests=requests))+'\n');f.flush();os.fsync(f.fileno())
def watch():
    while True:
        time.sleep(1)
        if pending and time.monotonic()-pending>90:os._exit(124)
threading.Thread(target=watch,daemon=True).start()
class Reader:
    def __init__(self,p):
        global pending
        guard();pending=time.monotonic();ledger()
        try:self.f=open(p,'rb')
        finally:pending=0
    def read(self,n=-1):
        global count,requests,pending
        assert n>=0 and count+n<3*1024**3;guard();count+=n;requests+=1;ledger();pending=time.monotonic()
        try:return self.f.read(n)
        finally:pending=0
    def seek(self,*a):
        global pending
        guard();pending=time.monotonic();ledger()
        try:return self.f.seek(*a)
        finally:pending=0
    def tell(self):return self.f.tell()
    def close(self):self.f.close()
seen=set();pixels=set();allallowed={'AIF005','AIF006','AIF007','AIF010','AIF012','AIF013'}
for source in sorted({r['source_path'] for r in final['rows']}):
    reader=Reader(source)
    try:
        with tarfile.open(fileobj=reader,mode='r:') as tf:
            for r in final['rows']:
                if r['source_path']!=source:continue
                assert r['farm_id'] in allallowed
                m=tf.getmember(r['source_member']);assert m.size==r['bytes']
                payload+=m.size;assert payload<2*1024**3;ledger()
                raw=tf.extractfile(m).read();saved=Path(r['final_path']).read_bytes()
                assert raw==saved and hashlib.sha256(saved).hexdigest()==r['sha256'] and r['sha256'] not in seen;seen.add(r['sha256'])
                with Image.open(io.BytesIO(saved)) as im:
                    assert list(im.size)==r['actual_size'];im=ImageOps.exif_transpose(im).convert('RGB');im.load()
                    pixels.add((im.size,hashlib.sha256(im.tobytes()).hexdigest()))
    finally:reader.close()
old=json.loads((RUN.parent/'image_similar_collection_20261010_v1/final_manifest_v1.json').read_text(encoding='utf-8'))
assert not seen & {r['sha256'] for r in old['images']}
oldpixels=set()
for r in old['images']:
    with Image.open(r['final_path']) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');oldpixels.add((im.size,hashlib.sha256(im.tobytes()).hexdigest()))
csvrows=list(csv.DictReader((Path(final['folder'])/'선정목록_v1.csv').open(encoding='utf-8-sig')))
assert len(csvrows)==len(final['rows'])==len(list(Path(final['folder']).glob('*.jpg')))==final['count']
for a,r in zip(csvrows,final['rows']):assert a['ID']==r['image_id'] and a['SHA256']==r['sha256'] and a['유사강도']==r['visual_review']['degree']
sql=sqlite3.connect(':memory:');sql.execute('create table x(kind text,farm text)');sql.executemany('insert into x values(?,?)',[(r['품종'],r['농장']) for r in csvrows])
assert dict(sql.execute('select kind,count(*) from x group by kind'))==final['kind_counts']
assert dict(sql.execute('select farm,count(*) from x group by farm'))==final['farm_counts']
photo_count=len({(r['source_archive'],r['source_member']) for r in scan['results']});assert photo_count==len(scan['results'])
out=dict(pass_all=True,count=final['count'],unique_jpeg_sha=len(seen),unique_decoded_pixels=len(pixels),cross_old_sha=0,cross_old_decoded_pixel_duplicates=len(pixels&oldpixels),independent_tarfile_byte_match=len(seen),kind_counts=final['kind_counts'],farm_counts=final['farm_counts'],new_examined_photos=photo_count,completed_plan_units=len(scan['completed_plan_units']),scan_errors=scan['errors'],read_requested_bytes_after_verification=count,payload_bytes_after_verification=payload,provider_hydration='not measured',scope='사전 선정 개발농장 일부 원본 검색, 전체350GB전수아님')
with (RUN/'verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
