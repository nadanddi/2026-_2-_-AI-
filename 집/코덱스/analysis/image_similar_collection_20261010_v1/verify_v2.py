from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,csv,hashlib,tarfile,sqlite3,collections
from PIL import Image
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
CACHE=ROOT/'집/코덱스/local/image_source_mapping_20261010_v1'
d=json.loads((RUN/'final_manifest_v1.json').read_text(encoding='utf-8'))
rows=d['images'];files=list((LOCAL/'선별한_유사사진_49장').glob('*.jpg'))
assert len(files)==len(rows)==49
seen=set()
for archive in {r['source_archive'] for r in rows}:
    with tarfile.open(CACHE/archive,'r:') as tf:
        for r in rows:
            if r['source_archive']!=archive:continue
            original=tf.extractfile(r['source_member']).read();raw=Path(r['final_path']).read_bytes()
            assert raw==original and hashlib.sha256(raw).hexdigest()==r['sha256']
            with Image.open(r['final_path']) as im:assert list(im.size)==r['size'];im.verify()
            assert r['sha256'] not in seen;seen.add(r['sha256'])
csvrows=list(csv.DictReader((LOCAL/'선별한_유사사진_49장/선정목록_v1.csv').open(encoding='utf-8-sig')))
assert len(csvrows)==49 and {r['ID'] for r in csvrows}=={r['image_id'] for r in rows}
for a,r in zip(csvrows,rows):
    assert a['SHA256']==r['sha256'] and a['품종']==r['kind'] and a['농장']==r['farm']
con=sqlite3.connect(':memory:');con.execute('create table x(id text,kind text,farm text,prefix text)')
con.executemany('insert into x values(?,?,?,?)',[(r['ID'],r['품종'],r['농장'],r['그룹접두사']) for r in csvrows])
assert dict(con.execute('select kind,count(*) from x group by kind'))==d['kind_counts']
assert dict(con.execute('select farm,count(*) from x group by farm'))==d['farm_counts']
assert dict(con.execute('select prefix,count(*) from x group by prefix'))==d['prefix_counts']
dec=list(csv.DictReader((RUN/'육안_선정판정_v1.csv').open(encoding='utf-8-sig')))
assert len(dec)==63 and sum(z['decision']=='제외' for z in dec)==8
assert len({z['image_id'] for z in dec if z['decision']=='유지'})==49
for p in d['boards']:
    with Image.open(p) as im:im.verify()
image_cache_bytes=sum(p.stat().st_size for task in (ROOT/'집/코덱스/local').glob('image_*') for p in task.rglob('*') if p.is_file())
out=dict(pass_all=True,fresh_jpeg_count=len(files),unique_sha=len(seen),independent_tarfile_byte_equality_count=len(rows),kind_counts=d['kind_counts'],farm_counts=d['farm_counts'],reference_decisions=len(dec),excluded_reference_pairs=8,excluded_unique_candidates=7,image_cache_bytes=image_cache_bytes,scope=d['search_scope'],limits=d['limits'])
assert image_cache_bytes<16*1024**3
with (RUN/'verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
