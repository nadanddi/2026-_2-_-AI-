"""Independent aggregate and copied-D2 verification, without new fitting."""
import csv,json,hashlib,math
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
table=read(HERE/'진단표_v1.csv');res=json.loads((HERE/'d1_result_v1.json').read_text(encoding='utf-8'))
source=ROOT/'연구실/코덱스/local/im_n9_error_diagnosis_20261016_v1/rows_v1.csv'
raw=read(source);miss=[r for r in raw if r['dataset']!='im_set_realg' and int(r['label'])==1 and float(r['p9'])<.5]
found=lambda r: all(r[k] and math.isfinite(float(r[k])) for k in ('dg1','gap')) and float(r['dg1'])<=1.0110872983932495 and float(r['gap'])>=3.4739584922790527
detail=[r for r in table if r['구분']=='D1 놓친 가짜 상세']
assert len(detail)==len(miss)==res['missed_fake']==181
lookup={(r['dataset'].replace('im_set_',''),r['file']):r for r in miss}
assert len(lookup)==181
for r in detail:
    old=lookup[r['자료'],r['파일']]
    assert r['kind']==old['kind'] and float(r['p9'])==float(old['p9']) and int(r['원본후보발견'])==int(found(old))
assert sum(map(found,miss))==sum(int(r['원본후보발견']) for r in detail)==res['found']==37
for kind,counts in res['by_kind'].items():
    rr=[r for r in miss if r['kind']==kind]
    assert counts==dict(n=len(rr),found=sum(map(found,rr)))
assert sum(int(r['rc']) for r in miss)==res['rc_count']==17
prior=read(ROOT/'연구실/코덱스/analysis/im_n9_error_diagnosis_20261016_v1/진단표_v2.csv')
d2=[r for r in table if r['구분'].startswith('D2 ')]
expected=[r for r in prior if r['구분'].startswith('②')]
assert len(d2)==len(expected)
for new,old in zip(d2,expected):
    assert all(new[k]==('D2 '+v if k=='구분' else v) for k,v in old.items())
assert len(table)==res['table_rows']==471 and len((HERE/'요약_v1.md').read_text(encoding='utf-8').splitlines())==5
out=dict(pass_all=True,table_rows=len(table),missed_fake=181,found=37,rate=37/181,branch='A2',d2_copied_rows=len(d2),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'진단표_v1.csv',HERE/'요약_v1.md')})
with (HERE/'verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
