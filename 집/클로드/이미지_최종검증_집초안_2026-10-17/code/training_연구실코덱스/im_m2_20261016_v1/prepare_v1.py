"""Freeze the single decision change and inventory the raw inputs without scoring."""
import csv, json, hashlib, difflib
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RAW=ROOT/'연구실/클로드/local/im_r5_raw'
LOCAL=ROOT/'연구실/코덱스/local/im_m2_20261016_v1'
src=ROOT/'연구실/클로드/image/im66_meta_v2.py'
text=src.read_text(encoding='utf-8-sig')
old='if un.mean() <= .005: break'
new='if un.mean() <= float(O9.mean()) + .001: break'
assert text.count(old)==1
newtext=text.replace(old,new)
oldimp='sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H'
newimp='sys.path.insert(0, str(Path.home() / "storage/nadan/farmai_img/data/image")); import im44_h_train_eval as H'
assert newtext.count(oldimp)==1
newtext=newtext.replace(oldimp,newimp)
key='L = H.L; NB = 2000; TAG = "m_v2"'
assert newtext.count(key)==1
newtext=newtext.replace(key,key+'\nimport os\nOUT = Path(os.environ["M2_STAGE_DIR"])')
assert newtext.count('L / f"preds/{TAG}')==3
newtext=newtext.replace('L / f"preds/{TAG}', 'OUT / f"{TAG}')
newtext=newtext.replace('== 合집합 예산 0.5% 초과', '== 合집합 예산 O9+0.1%p 초과')
# Reverse all permitted edits; exact recovery proves no further source changes.
back=newtext.replace(new,old).replace(newimp,oldimp).replace(key+'\nimport os\nOUT = Path(os.environ["M2_STAGE_DIR"])',key).replace('OUT / f"{TAG}','L / f"preds/{TAG}').replace('== 合집합 예산 O9+0.1%p 초과','== 合집합 예산 0.5% 초과')
assert back==text
target=HERE/'im66_m2_v1.py'
with target.open('x',encoding='utf-8',newline='\n') as f: f.write(newtext)
with (HERE/'source_diff_v1.patch').open('x',encoding='utf-8') as f:
    f.writelines(difflib.unified_diff(text.splitlines(True),newtext.splitlines(True),fromfile='im66_meta_v2.py',tofile='im66_m2_v1.py'))
rr=list(csv.DictReader((RAW/'manifest.csv').open(encoding='utf-8-sig')))
assert len(rr)==700 and all(r['role']=='hold3' for r in rr)
assert len({r['file'] for r in rr})==700
records=[]
for r in rr:
    p=(RAW/r['tar']/r['file']).resolve()
    assert p.is_relative_to(RAW.resolve()) and p.is_file()
    records.append(dict(path=p.relative_to(RAW).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
with (LOCAL/'raw_inventory_v1.json').open('x',encoding='utf-8') as f: json.dump(records,f,ensure_ascii=False,indent=2)
meta=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),copy_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
          semantic_change=new,exact_reverse_check=True,raw_images=len(rr),raw_groups=len({r['group'] for r in rr}),raw_bytes=sum(r['bytes'] for r in records),
          manifest_sha256=hashlib.sha256((RAW/'manifest.csv').read_bytes()).hexdigest())
with (HERE/'registration_v1.json').open('x',encoding='utf-8') as f: json.dump(meta,f,ensure_ascii=False,indent=2)
print(json.dumps(meta,ensure_ascii=False))
