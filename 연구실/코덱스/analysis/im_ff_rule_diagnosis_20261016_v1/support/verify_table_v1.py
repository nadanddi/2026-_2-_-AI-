"""Cross-check every standalone-rule dataset row and its FP denominator."""
import csv,json,tarfile,io
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[3]
T=json.loads((P/'support/config_v2.json').read_text(encoding='utf-8'))['finefp'];D={}
with tarfile.open(ROOT/'연구실/코덱스/local/im_ff_rule_diagnosis_20261016_v1/raw_v2.tar') as tf:
    for m in tf.getmembers():
        if m.name.endswith('.npz'):
            with np.load(io.BytesIO(tf.extractfile(m).read())) as z:D[Path(m.name).stem]={k:z[k] for k in z.files}
with (P/'진단표_v1.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
checks=0;rule_rows=0;kind_rows=0
for r in rows:
    if r['구분'] not in ('E3 자료별경보','E3 자료유형별부분편집'):continue
    z=D[r['자료']];f=z['ff'];vals=dict(dg1=f[:,0],gap=f[:,1],lg=f[:,2],le=f[:,3],abs_ge=abs(f[:,4]),lg_div_absge_floor1=f[:,2]/np.maximum(abs(f[:,4]),1),le_div_absge_floor1=f[:,3]/np.maximum(abs(f[:,4]),1))
    a=np.isfinite(f).all(1)&(vals[r['항목']]>=float(r['값']))
    if r['집단']=='ff유효_gap기존이상':a&=f[:,1]>=T['gap']
    y=z['y'];b=z['p9']>=.5;c=b|a
    if r['구분']=='E3 자료유형별부분편집':
        sel=(y==1)&~b&(z['kind']==r['kind']);assert int(r['수'])==int((a&sel).sum()) and int(r['분모'])==int(sel.sum());checks+=2;kind_rows+=1;continue
    expected={'수':int(a.sum()),'분모':len(a),'고친가짜':int(((y==1)&~b&c).sum()),'새진짜오판':int(((y==0)&~b&c).sum()),'기존진짜오판':int(((y==0)&b).sum()),'반사실진짜오판':int(((y==0)&c).sum()),'진짜분모':int((y==0).sum()),'진짜경보':int(((y==0)&a).sum())}
    for k,v in expected.items():assert int(r[k])==v,(r['자료'],r['항목'],k);checks+=1
    if (y==0).any():
        assert abs(float(r['진짜경보율'])-((y==0)&a).sum()/(y==0).sum())<1e-15
        assert abs(float(r['반사실오탐률'])-((y==0)&c).sum()/(y==0).sum())<1e-15
        checks+=2
    rule_rows+=1
assert rule_rows==238 and kind_rows==126
with (P/'support/verification_table_v1.json').open('x') as f:json.dump(dict(pass_all=True,checks=checks,rule_dataset_rows=rule_rows,rule_dataset_kind_rows=kind_rows),f,indent=2)
print('TABLE_VERIFIED',checks,rule_rows,kind_rows)
