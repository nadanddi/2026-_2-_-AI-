import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,io,hashlib,collections,shutil
from PIL import Image,ImageOps,ImageDraw,ImageFont
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
selection=json.loads((RUN/'selection_v1.json').read_text(encoding='utf-8'))
tainted={r['prefix_candidate'] for r in selection['candidates'] if any(z in r['excluded_reasons'] for z in ['near_public_phash_candidate','public_exact_byte','prior_public_scene_family_suspect'])}
chosen=[];families=set()
for rank in selection['rankings'][:3]:
    for row in rank['top5']:
        if row['prefix_candidate'] not in families|tainted:
            chosen.append(dict(row,selection_reference=rank['public_file']));families.add(row['prefix_candidate']);break
assert len(chosen)>=2
prepared=[]
for i,row in enumerate(chosen[:2]):
    cache=ROOT/'집/코덱스/local/image_source_mapping_20261010_v1'/row['source_archive']
    with cache.open('rb') as f:f.seek(row['offset']);raw=f.read(row['bytes'])
    assert hashlib.sha256(raw).hexdigest()==row['payload_sha256']
    prior_bytes=sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file()); reserve=len(raw)+8*1024**2
    assert prior_bytes+reserve<256*1024**2 and shutil.disk_usage(ROOT).free-reserve>30*1024**3
    src=LOCAL/f'source_{i:02}_original_v1.jpg';src.open('xb').write(raw)
    with Image.open(io.BytesIO(raw)) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.load()
    im.thumbnail((768,768),Image.Resampling.LANCZOS)
    target=LOCAL/f'source_{i:02}_edit_target_v1.png';assert not target.exists();im.save(target)
    row.update(original_path=str(src),target_path=str(target),target_size=list(im.size),target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),magenta_rule='R-G>20 and B-G>20; reject fraction>0.35',public_observation_used=True)
    prepared.append(row)
size=sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file());assert size<256*1024**2 and shutil.disk_usage(ROOT).free>30*1024**3
with (RUN/'prepared_selection_v1.json').open('x',encoding='utf-8') as f:json.dump({'sources':prepared,'tainted_prefix_families':sorted(tainted),'selection_scope':'first2 distinct eligible prefixes from real3 top5; no independent scene authentication','task_bytes':size},f,ensure_ascii=False,indent=2)
print(json.dumps([(r['image_id'],r['target_path'],r['selection_reference']) for r in prepared],ensure_ascii=False))
