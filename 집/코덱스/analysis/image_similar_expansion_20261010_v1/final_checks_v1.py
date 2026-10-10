from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,csv,collections,sqlite3
from PIL import Image,ImageOps
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
v=json.loads((RUN/'verification_v1.json').read_text(encoding='utf-8'))
assert v['pass_all'] and v['count']==v['unique_jpeg_sha']==v['unique_decoded_pixels']==55
assert v['cross_old_sha']==v['cross_old_decoded_pixel_duplicates']==v['cross_public_decoded_pixel_duplicates']==0
combined=json.loads((RUN/'combined_manifest_v1.json').read_text(encoding='utf-8'))
csvrows=list(csv.DictReader((Path(combined['folder'])/'사진별_선정이유_v1.csv').open(encoding='utf-8-sig')))
assert len(csvrows)==len(combined['rows'])==len(list(Path(combined['folder']).glob('*.jpg')))==104
seen=set();pix=set()
for a,r in zip(csvrows,combined['rows']):
    raw=Path(r['path']).read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==r['sha256']==a['SHA256'] and sha not in seen;seen.add(sha)
    assert a['유사강도']==r['degree'] and a['공개참조']==r['reference'] and a['선정이유']==r['reason']
    with Image.open(r['path']) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');pix.add((im.size,hashlib.sha256(im.tobytes()).hexdigest()))
assert len(pix)==104
sql=sqlite3.connect(':memory:');sql.execute('create table x(kind text,farm text,phase text)');sql.executemany('insert into x values(?,?,?)',[(r['품종'],r['농장'],r['선정차수']) for r in csvrows])
assert dict(sql.execute('select kind,count(*) from x group by kind'))==combined['kind_counts']
assert dict(sql.execute('select farm,count(*) from x group by farm'))==combined['farm_counts']
assert dict(sql.execute('select phase,count(*) from x group by phase'))=={'1차49':49,'추가55':55}
scan=json.loads((RUN/'scan_expanded_v3.json').read_text(encoding='utf-8'));prior=json.loads((RUN/'scan_expanded_v2.json').read_text(encoding='utf-8'))
assert len(scan['results'])==24*len(scan['completed_plan_units'])==408
assert len({r['farm_id'] for r in scan['results']})==6 and len({r['source_archive'] for r in scan['results']})==15
known={r['payload_sha256'] for r in json.loads((RUN.parent/'image_source_mapping_20261010_v1/mapping_v1.json').read_text(encoding='utf-8'))['mapped']}
assert not {r['sha256'] for r in scan['results']} & known
assert len({(r['source_archive'],r['source_member']) for r in scan['results']})==408
fresh=json.loads((RUN/'final_manifest_v1.json').read_text(encoding='utf-8'));newcsv=list(csv.DictReader((Path(fresh['folder'])/'선정목록_v1.csv').open(encoding='utf-8-sig')))
for a,r in zip(newcsv,fresh['rows']):assert a['비교샘플']==r['visual_review']['reference'] and a['선정이유']==r['visual_review']['reason'] and a['그룹접두사']==r['prefix_candidate']
for i in range(1,8):
    a=LOCAL/('공개샘플_사진별_짝비교_%02d_v1.jpg'%i);b=LOCAL/('공개샘플_사진별_짝비교_%02d_v2.jpg'%i)
    assert a.read_bytes()==b.read_bytes()
    with Image.open(b) as im:im.verify()
with Image.open(LOCAL/'공개샘플_사진별_짝비교_04_v2.jpg') as im:im.crop((800,42,1200,342)).save(LOCAL/'짝비교04_상단우측_확인_v1.png')
cache=sum(p.stat().st_size for task in (ROOT/'집/코덱스/local').glob('image_*') for p in task.rglob('*') if p.is_file());own=sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())
assert cache<16*1024**3 and own<2*1024**3
out=dict(pass_all=True,combined_count=104,combined_unique_jpeg_sha=104,combined_unique_decoded_pixels=104,combined_kind_counts=combined['kind_counts'],new_count=55,examined_new=408,source_farms=6,source_tar_files=15,source_plan_units=17,source_plan_total_units=24,prior_stage_examined=len(prior['results']),prior_stage_stop_reasons=prior['errors'],coverage_additional_examined=len(scan['results'])-len(prior['results']),paired_rerender_byte_equal_count=7,manual_reference_csv_verified=True,image_cache_bytes=cache,new_task_bytes=own,limits='104は独立場面数ではない')
out['limits']='104は' if False else '104는 독립 장면 수가 아님. 전체350GB/24묶음576전수완료 아님.'
with (RUN/'final_checks_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
