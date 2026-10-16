"""Score all tokens, OOF calibration and one fixed A2 evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from tokens_v1 import Tokens,L,read
from eval_support_v1 import H,metadata,gate,n9
HERE=Path(__file__).resolve().parent
def dump(name,v):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def sigmoid(z):return 1/(1+np.exp(-np.clip(np.asarray(z,dtype=np.float64),-700,700)))
models=[]
for i in range(6):
    with np.load(HERE/(f'fold{i}_model_v1.npz' if i<5 else 'final_model_v1.npz')) as z:models.append({k:z[k] for k in z.files})
W=np.stack([m['coef']/m['scale'] for m in models],axis=1).astype(np.float32)
b=np.array([m['intercept'][0]-m['mean']@(m['coef']/m['scale']) for m in models],np.float32)
m=Tokens();rows=json.loads((HERE/'photos_v1.json').read_text(encoding='utf-8'))
dump('score_alignment_v1.json',m.alignment(L/rows[0]['dataset']/'img'/rows[0]['file']))
W=torch.from_numpy(W).to(m.dev);b=torch.from_numpy(b).to(m.dev)
with np.load(HERE/'samples_v1.npz') as z:fold=z['fold']
pool=[];cal_files=[]
for i,r in enumerate(rows):
    if int(r['label'])!=0:continue
    p=L/r['dataset']/'img'/r['file']
    if not gate(p):continue
    v=float(sigmoid(m.max_logits(p,W,b)[fold[i]]));pool.append(v);cal_files.append(r['dataset']+'/'+r['file'])
    if len(pool)%200==0:print('A2_OOF_SCORE',len(pool),flush=True)
assert len(pool)==1800,len(pool)
SETS=('im_set_newreal','im_set_realg','im_set_realn','im_set_r2','im_set_r3','im_set_r4','im_set_r5','sealed_real','sealed_brown','sealed_bright','jig500','pub21','im_set_r3p_sharp','im_set_r3p_area','im_set_r3p_png','im_set_r3p_webp','im_set_r3p_c444','im_set_gpt1','im_set_gpt3','im_set_gptr','im_set_gptr_src')
scored={}
for sn in SETS:
    F=H.load(sn);rr,ps=metadata(sn);assert F['ok'].all() and F['files'].tolist()==[r['file'] for r in rr]
    if sn in ('im_set_gptr','im_set_gptr_src'):
        keep=np.array([r['split']=='hold' for r in rr]);F={k:v[keep] for k,v in F.items()}
        rr=[r for r,k in zip(rr,keep) if k];ps=[p for p,k in zip(ps,keep) if k]
    gg=np.array([gate(p) for p in ps]);y=np.array([int(r['label']) for r in rr]);scores=np.zeros(len(rr))
    for i,p in enumerate(ps):
        if gg[i]:scores[i]=sigmoid(m.max_logits(p,W,b)[5])
        if (i+1)%200==0:print('A2_SCORE',sn,i+1,'/',len(rr),flush=True)
    base=np.where(gg,n9(F),F['p6']);assert np.isfinite(scores).all()
    ref=HERE.parents[1]/'im_meta_probe_20261016_v1'/f'pred_1601_{sn}.csv'
    if sn!='im_set_r5':
        assert ref.is_file(),ref
        br=read(ref);assert [r['file'] for r in br]==[r['file'] for r in rr]
        assert np.array_equal(y,[int(r['label']) for r in br])
        assert np.allclose(base,[float(r['p9']) for r in br],rtol=0,atol=1e-12)
    if sn=='im_set_r5':
        with np.load(HERE.parents[1]/'im_m2_20261016_v1/open/eval_im_set_r5.npz') as z:
            assert np.array_equal(y,z['y']) and np.array_equal([r['kind'] for r in rr],z['kind'])
            assert np.array_equal([r['group'] for r in rr],z['group'])
            assert np.allclose(base,z['p9'],rtol=0,atol=1e-12)
    np.savez(HERE/f'score_{sn}.npz',files=F['files'],y=y,kind=np.array([r.get('kind','unknown') for r in rr]),base=base,score=scores,gate=gg)
    scored[sn]=(y,base,scores,gg)
    if sn in SETS[:3]:pool.extend(scores[gg].tolist());cal_files.extend(sn+'/'+r['file'] for r,g in zip(rr,gg) if g)
P=np.array(pool);assert len(P)==3195 and np.isfinite(P).all();theta=float(np.quantile(P,.9975));assert 0<theta<1,theta
np.savez(HERE/'calibration_v1.npz',scores=P,files=np.array(cal_files))
dump('theta_v1.json',dict(theta=theta,pool=len(P),oof=1800,final_calibration=1395))
E={}
for sn,(y,base,scores,gg) in scored.items():
    candidate=np.where(gg,np.maximum(base,H.g(scores,theta)),base);assert np.array_equal(candidate[~gg],base[~gg])
    np.savez(HERE/f'eval_{sn}.npz',y=y,base=base,candidate=candidate,gate=gg)
    ba=lambda p:float(.5*((p[y==0]<.5).mean()+(p[y==1]>=.5).mean())) if (y==0).any() and (y==1).any() else None
    E[sn]=dict(n=len(y),base_fp=int(((y==0)&(base>=.5)).sum()),candidate_fp=int(((y==0)&(candidate>=.5)).sum()),new_fp=int(((y==0)&(base<.5)&(candidate>=.5)).sum()),fixed_fake=int(((y==1)&(base<.5)&(candidate>=.5)).sum()),base_ba=ba(base),candidate_ba=ba(candidate),correct=int(((candidate>=.5)==y).sum()))
    print('A2_EVAL',sn,E[sn],flush=True)
core=SETS[3:7];guards=('sealed_real','sealed_brown','sealed_bright')
conditions=dict(fixes_exceed_fp=sum(E[s]['fixed_fake'] for s in core)>sum(E[s]['new_fp'] for s in core),sealed_new_fp=sum(E[s]['new_fp'] for s in guards)<=1,jig_delta=(E['jig500']['candidate_fp']-E['jig500']['base_fp'])/E['jig500']['n']<=.01,public21=E['pub21']['correct']==21,r345_ba=all(E[s]['candidate_ba']>=E[s]['base_ba'] for s in core[1:]))
dump('result_v1.json',dict(theta=theta,metrics=E,conditions=conditions,internal_pass=all(conditions.values())))
print('A2_FINAL',conditions,'PASS',all(conditions.values()),flush=True)
