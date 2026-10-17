"""Independent array-based reproduction of the frozen M2 rules; stage is explicit."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
DATA=ROOT/'연구실/코덱스/local/im_m2_20261016_v1'
stage=sys.argv[1]; assert stage in ('preflight','open')
D=DATA/stage
if stage=='open': assert (DATA/'R5_OPEN_ONCE.json').exists()
state=json.loads((D/'state_v1.json').read_text())
C={}
def ba(y,p):
    return .5*(np.count_nonzero(p[y==0]<.5)/np.count_nonzero(y==0)+np.count_nonzero(p[y==1]>=.5)/np.count_nonzero(y==1))
for f in sorted(D.glob('eval_*.npz')):
    key=f.stem[5:]
    assert key!='im_set_r5' or stage=='open'
    z=dict(np.load(f)); y,b,p=z['y'],z['p9'],z['pm']
    assert np.isfinite(b).all() and np.isfinite(p).all()
    assert ((b>=0)&(b<=1)&(p>=0)&(p<=1)).all()
    assert (p>=b).all(),'additive monotonicity'
    z['nfp']=int(np.count_nonzero((b<.5)&(p>=.5)&(y==0)))
    z['fx']=int(np.count_nonzero((b<.5)&(p>=.5)&(y==1)))
    z['ba9']=ba(y,b) if (y==0).any() and (y==1).any() else None
    z['bam']=ba(y,p) if (y==0).any() and (y==1).any() else None
    C[key]=z
cal=np.load(D/'calibration_v1.npz'); P,rc,O9=cal['P'],cal['PRC'],cal['O9']
limit=O9.mean()+.001; attempts=[]
for fl,frc in ((.0025,.001),(.0015,.00075),(.001,.0005),(.0005,.0005)):
    th=float(np.quantile(P,1-fl)); thr=float(np.quantile(P[rc],1-frc)); union=O9|(P>=np.where(rc,thr,th))
    attempts.append(dict(flip=fl,flip_rc=frc,theta=th,theta_rc=thr,union_count=int(union.sum()),union_rate=float(union.mean()),pass_budget=bool(union.mean()<=limit)))
    if union.mean()<=limit: break
chosen=attempts[-1]
assert np.array_equal(union,cal['un'])
assert th==state['budget']['theta'] and thr==state['budget']['theta_rc']
gates={}
if C:
    gates['g1']=C['im_set_r2']['bam']>=C['im_set_r2']['ba9']
    gates['g2']=all(C[s]['bam']>=C[s]['ba9'] for s in ('im_set_r3','im_set_r4'))
    def fp(s,key):
        q=C[s]; y=q['y']; return np.count_nonzero(q[key][y==0]>=.5)/np.count_nonzero(y==0)
    gates['g3']=(sum(C[s]['nfp'] for s in ('sealed_real','sealed_brown','sealed_bright'))<=2
        and fp('jig500','pm')<=fp('jig500','p9')+.01
        and all(fp('im_set_r3p_'+k,'pm')<=fp('im_set_r3p_'+k,'p9')+.01 for k in ('sharp','area'))
        and int(np.count_nonzero((C['pub21']['pm']>=.5)==C['pub21']['y']))==21
        and C['gptr_src_hold']['nfp']<=1)
    g5=np.load(D/'g5_v1.npz'); gates['g5']=g5['sM'].mean()<=g5['s9'].mean()+.005
    assert {k:bool(v) for k,v in gates.items()}==state['gates']
result=dict(stage=stage,budget=dict(n=len(P),o9_count=int(O9.sum()),o9_rate=float(O9.mean()),limit=float(limit),attempts=attempts),
    gates={k:bool(v) for k,v in gates.items()},sealed_real_new_fp=sum(C[s]['nfp'] for s in ('sealed_real','sealed_brown','sealed_bright')) if C else None,
    metrics={s:dict(n=len(q['y']),new_fp=q['nfp'],fixed_fake=q['fx'],ba9=q['ba9'],bam=q['bam']) for s,q in C.items()},
    all_finite_probability_monotonic=True,rows=sum(len(q['y']) for q in C.values()))
if stage=='open':
    q=C['im_set_r5']; gs=np.array(sorted(set(q['group']))); gi={x:np.where(q['group']==x)[0] for x in gs}
    rng=np.random.default_rng(66); dif=[]
    for _ in range(2000):
        ix=np.concatenate([gi[x] for x in rng.choice(gs,len(gs))]); y=q['y'][ix]
        if (y==0).any() and (y==1).any(): dif.append(ba(y,q['pm'][ix])-ba(y,q['p9'][ix]))
    pw=float((np.array(dif)<=0).mean())
    assert pw==state['r5_p_worse']
    cond=[q['bam']>q['ba9'],pw<.025,q['fx']>q['nfp'],q['nfp']<=1]
    assert cond==state['r5_conditions']
    result['r5']=dict(groups=len(gs),p_worse=pw,valid_bootstrap=len(dif),conditions=[bool(v) for v in cond],on_time=state['on_time'],adoption_candidate=bool(all(cond) and state['on_time']))
with (HERE/f'verification_{stage}_v1.json').open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
