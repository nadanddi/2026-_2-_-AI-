"""Independent standard-library counts/quartiles and original-mask checks."""
import csv,json,math,re,statistics,hashlib,tarfile,io
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent; RUN=HERE.parent; ROOT=HERE.parents[4]
D=ROOT/'연구실/코덱스/local/im_n9_error_diagnosis_20261016_v1'
def read(p):
    with p.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))
raw=read(D/'rows_v1.csv'); table=read(RUN/'진단표_v2.csv')
core=[r for r in raw if r['dataset']!='im_set_realg' and r['label']=='0']
high=[r for r in core if float(r['p9'])>=.6]; correct=[r for r in core if float(r['p9'])<.5]
assert len(high)==40 and len(correct)==2938 and sum(.5<=float(r['p9'])<.6 for r in core)==24
r5=[r for r in raw if r['dataset']=='im_set_r5']; errors=[r for r in r5 if (float(r['p9'])>=.5)!=(r['label']=='1')]
assert len(errors)==39
detail=[r for r in table if r['구분']=='① R5 오답 상세']
assert len(detail)==39 and {r['파일'] for r in detail}=={r['file'] for r in errors}
lookup={(r['dataset'].replace('im_set_',''),r['file']):r for r in raw}
counts=0
for r in table:
    if r['항목']=='개별 사진':
        src=lookup[r['자료'],r['파일']]
        for a,b in [('p9','p9'),('p6','p6'),('게이트','gate'),('rc','rc')]: assert r[a]==src[b]; counts+=1
    if r['구분']=='① R5 유형 집계':
        gg=[v for v in r5 if v['kind']==r['kind']]
        n=sum((float(v['p9'])>=.5)!=(v['label']=='1') for v in gg)
        assert int(r['표본수'])==n and int(r['분모'])==len(gg); counts+=2
def qt(v,q):
    s=sorted(v); x=(len(s)-1)*q; i=int(x); t=x-i
    return s[i] if i+1==len(s) else s[i]*(1-t)+s[i+1]*t
for r in table:
    if r['구분'] not in ('② 真짜 비교 통계','② 진짜 비교 통계','② 농가·품종 분포'): continue
    group=high if r['비교집단']=='진짜 오판 p≥.6' else correct
    group=[v for v in group if r['자료']=='전체' or v['dataset'].replace('im_set_','')==r['자료']]
    assert len(group)==int(r['분모'])
    if r['구분']=='② 진짜 비교 통계':
        assert int(r['표본수'])==len(group)
        assert r['rc']==f"{sum(int(v['rc']) for v in group)}/{len(group)}"
        for col,key in [('밝기','brightness'),('채도','saturation'),('갈변대리비율','brown'),('dg1','dg1'),('gap','gap')]:
            vals=[float(v[key]) for v in group if v[key]!='' and math.isfinite(float(v[key]))]
            nums=list(map(float,re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',r[col])))
            assert len(nums)==3
            expected=[statistics.median(vals),qt(vals,.25),qt(vals,.75)]
            assert all(abs(a-b)<=.00000050001 for a,b in zip(nums,expected)),(r['자료'],col,nums,expected)
            counts+=3
    else:
        assert abs(float(r['값'])-int(r['표본수'])/len(group))<1e-12
        counts+=1
# Every distribution partitions the full comparison group, including any unknowns.
groups={ (r['자료'],r['비교집단'],r['항목']) for r in table if r['구분']=='② 농가·품종 분포'}
for key in groups:
    sub=[r for r in table if r['구분']=='② 농가·품종 분포' and (r['자료'],r['비교집단'],r['항목'])==key]
    assert sum(int(r['표본수']) for r in sub)==int(sub[0]['분모']); counts+=1
with tarfile.open(ROOT/'연구실/코덱스/local/im_m2_20261016_v1/r5_artifacts_v1.tar') as tf:
    for r in detail:
        if r['kind'].startswith('partial'):
            name='im_set_r5/mask/'+Path(r['파일']).with_suffix('.png').name
            with Image.open(io.BytesIO(tf.extractfile(name).read())) as im: a=np.asarray(im.convert('L'))
            assert abs(float(r['마스크비율'])-float((a>127.5).mean()))<1e-15; counts+=1
        else: assert r['마스크비율']=='NA'
for r in raw:
    if r['dataset']=='im_set_realg' and r['Ef_fire']=='1':
        p=ROOT/'연구실/클로드/local/im_set_realg/img'/r['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['image_sha256']
        with Image.open(p) as im: rgb=np.asarray(im.convert('RGB')).astype(np.float64)
        expected32=float(((rgb.astype(np.float32)[...,0]+rgb.astype(np.float32)[...,1]+rgb.astype(np.float32)[...,2])/3).mean())
        # Cross-platform NumPy float32 reductions need not be bit-identical; use the independent float64 error bound.
        bound=np.finfo(np.float32).eps*255*(math.ceil(math.log2(rgb.shape[0]*rgb.shape[1]))+4)
        assert abs(rgb.mean()-float(r['brightness']))<=bound
        R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]; lum=rgb.mean(axis=2)
        assert float(((R>G)&(G-B>8)&(lum>50)&(lum<200)).mean())==float(r['brown'])
        counts+=3
rep=json.loads((HERE/'gpath_replay_v1.json').read_text(encoding='utf-8'))
assert len(rep['checks'])==5 and all(r['same_jpeg_bytes'] for r in rep['checks'])
assert len((RUN/'요약_v1.md').read_text(encoding='utf-8-sig').splitlines())==5
assert len(list(RUN.glob('진단표_*.csv')))==1
result=dict(checks=counts,table_rows=len(table),r5_errors=39,high_confidence_false_real=40,correct_real=2938,mid_false_real=24,
    gpath_replay_exact=5,one_current_table=True,summary_lines=5,fit_count=0,adoption_decision=False,
    table_sha256=hashlib.sha256((RUN/'진단표_v2.csv').read_bytes()).hexdigest())
with (HERE/'verification_final_v3.json').open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
