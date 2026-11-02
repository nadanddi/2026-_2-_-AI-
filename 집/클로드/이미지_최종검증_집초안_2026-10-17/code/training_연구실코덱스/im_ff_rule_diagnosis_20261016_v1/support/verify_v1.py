"""Independent checks of E1/E2/E3 table against raw cached arrays."""
import io,tarfile,json,csv,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent.parent;ROOT=HERE.parents[3]
data=ROOT/'연구실/코덱스/local/im_ff_rule_diagnosis_20261016_v1/raw_v2.tar'
R=json.loads((HERE/'support/result_v1.json').read_text(encoding='utf-8'));T=json.loads((HERE/'support/config_v2.json').read_text(encoding='utf-8'))['finefp'];D={};checks=0
with tarfile.open(data) as tf:
    for m in tf.getmembers():
        if m.name.endswith('.npz'):
            with np.load(io.BytesIO(tf.extractfile(m).read())) as z:D[Path(m.name).stem]={k:z[k] for k in z.files}
core=('im_set_r2','im_set_r3','im_set_r4','im_set_r5');high=0;onlypcap=0;rc_already=0
for sn,z in D.items():
    f=z['ff'];v=np.isfinite(f).all(1)
    safe=v&(f[:,0]<=T['m'])&(f[:,1]>=T['gap'])&(f[:,2]<=T['r_g'])&(f[:,3]<=T['r_e'])&(abs(f[:,4])<=T['r_glob'])&~((f[:,2]>T['f_g'])|(f[:,3]>T['f_e']))
    assert np.array_equal(safe,z['rc_no_pcap']);assert np.array_equal(safe&(z['ppre']<.8),z['rc']);checks+=2
    yy=z['y'];bb=z['p9']>=.5;cc=z['p9_no_pcap']>=.5;old=R['E2'][sn]
    calc={'고친진짜':int(((yy==0)&bb&~cc).sum()),'새가짜놓침':int(((yy==1)&bb&~cc).sum()),'새진짜오판':int(((yy==0)&~bb&cc).sum()),'고친가짜':int(((yy==1)&~bb&cc).sum())}
    assert all(calc[k]==old[k] for k in calc);checks+=4
    if sn in core:
        h=(yy==0)&(z['p9']>=.6);high+=int(h.sum());onlypcap+=int((h&safe&~z['rc']).sum());rc_already+=int((h&z['rc']).sum())
assert (high,onlypcap,rc_already)==(40,10,11);checks+=3
cal=[D[s] for s in ('im_set_newreal','im_set_realg','im_set_realn')]
def score(z,name):
    f=z['ff']
    return {'dg1':f[:,0],'gap':f[:,1],'lg':f[:,2],'le':f[:,3],'abs_ge':abs(f[:,4]),'lg_div_absge_floor1':f[:,2]/np.maximum(abs(f[:,4]),1),'le_div_absge_floor1':f[:,3]/np.maximum(abs(f[:,4]),1)}[name]
def eligible(z,scope):return np.isfinite(z['ff']).all(1)&((z['ff'][:,1]>=T['gap']) if scope.endswith('gap기존이상') else True)
for key,rec in R['E3'].items():
    scope,name=key.split('|');th=rec['threshold'];cal_count=sum(int((eligible(z,scope)&(score(z,name)>=th)).sum()) for z in cal)
    assert cal_count==rec['calibration_alerts'] and cal_count<=3;checks+=2
    caught=0
    for sn in core:
        z=D[sn];target=(z['y']==1)&(z['p9']<.5)&np.isin(z['kind'],('partial_style','partial_gen_v7','partial_gen_v7s'))
        caught+=int((target&eligible(z,scope)&(score(z,name)>=th)).sum())
    assert caught==rec['caught147'];checks+=1
with (HERE/'진단표_v1.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==R['table_rows'];assert len((HERE/'요약_v1.md').read_text(encoding='utf-8').splitlines())==5
assert sum(r['구분']=='E1 개별40' for r in rows)==40
out=dict(pass_all=True,checks=checks,table_rows=len(rows),raw_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),E4_access=False,adoption_or_submission=False)
with (HERE/'support/verification_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
