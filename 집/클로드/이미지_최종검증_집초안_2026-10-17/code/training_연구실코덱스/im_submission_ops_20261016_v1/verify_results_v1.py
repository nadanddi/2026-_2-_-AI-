"""Independent final metrics from frozen per-photo arrays."""
import sys,json
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);mode=sys.argv[2];D=root/mode
result=json.loads((D/'result_v1.json').read_text(encoding='utf-8'));E={};checks=0;rows=0
for sn,old in result['metrics'].items():
    with np.load(D/f'eval_{sn}.npz') as z:
        y=z['y'];base=z['base'];cand=z['candidate'];gg=z['gate']
        assert len(y)==old['n'] and all(a.shape==y.shape for a in (base,cand,gg))
        assert np.isfinite(base).all() and np.isfinite(cand).all() and ((0<=cand)&(cand<=1)).all()
        assert np.array_equal(base[~gg],cand[~gg])
        real=y==0;fake=y==1;bp=base>=.5;cp=cand>=.5
        ba=lambda p:float(.5*((~p[real]).sum()/real.sum()+p[fake].sum()/fake.sum())) if real.any() and fake.any() else None
        M=dict(n=len(y),base_fp=int((real&bp).sum()),candidate_fp=int((real&cp).sum()),new_fp=int((real&~bp&cp).sum()),base_ba=ba(bp),candidate_ba=ba(cp),correct=int((cp==y).sum()))
        if mode=='c1':
            M['filter_losses']=int((fake&np.array([str(k).startswith('filter_') for k in z['kind']])&bp&~cp).sum())
        else:
            assert (cand>=base).all()
            M['fixed_fake']=int((fake&~bp&cp).sum())
        for k,v in M.items():
            assert k in old
            if v is None:assert old[k] is None
            elif isinstance(v,float):assert abs(v-old[k])<1e-15,(sn,k,v,old[k])
            else:assert v==old[k],(sn,k,v,old[k])
            checks+=1
        E[sn]=M;rows+=len(y)
core=('im_set_r2','im_set_r3','im_set_r4','im_set_r5');guards=('sealed_real','sealed_brown','sealed_bright')
if mode=='c1':
    U=('im_set_newreal','im_set_realg','im_set_realn')
    conditions=dict(real_fp_reduction=sum(E[s]['base_fp']-E[s]['candidate_fp'] for s in U)>=3,filter_losses=sum(E[s]['filter_losses'] for s in core)<=2,r345_ba=all(E[s]['candidate_ba']>=E[s]['base_ba'] for s in core[1:]),public21=E['pub21']['correct']==21,real_guards=all(E[s]['new_fp']==0 for s in ('jig500',*guards)))
else:
    conditions=dict(fixes_exceed_fp=sum(E[s]['fixed_fake'] for s in core)>sum(E[s]['new_fp'] for s in core),sealed_new_fp=sum(E[s]['new_fp'] for s in guards)<=1,jig_delta=(E['jig500']['candidate_fp']-E['jig500']['base_fp'])/E['jig500']['n']<=.01,public21=E['pub21']['correct']==21,r345_ba=all(E[s]['candidate_ba']>=E[s]['base_ba'] for s in core[1:]))
assert conditions==result['conditions'] and all(conditions.values())==result['internal_pass']
out=dict(verification_pass=True,arrays=len(E),rows=rows,metric_checks=checks,conditions=conditions,candidate_internal_pass=all(conditions.values()))
with (D/'verification_final_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False),flush=True)
