from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import numpy as np,pandas as pd
P=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv'
oof=pd.read_csv(P,float_precision='round_trip');z=oof[(oof.validator=='DIAG10')&(oof.seed==7)]
support=pd.read_csv(H/'summary_v1/smooth_day_support_profiles.csv',float_precision='round_trip');out=[]
for context,farm,day,fold in [('actual0_7','F47',160,0),('actual1_7','F47',161,1),('actual0_7','F13',112,0)]:
    q=z[(z.farm==farm)&(z.day==day)&(z.fold==fold)].sort_values('hour');assert len(q)==24
    smooth=lambda a:.5*a+.5*np.cumsum(a)/np.arange(1,25)
    et=smooth(q.raw_et.to_numpy());lp=smooth(q.raw_lgb.to_numpy());mp=smooth(q.raw_mlp.to_numpy());pp=smooth(q.old_pfn_raw.to_numpy())
    blend=.48*et+.24*lp+.08*mp+.2*pp;assert np.max(abs(np.clip(blend,q.clip_lo.to_numpy(),q.clip_hi.to_numpy())-q.baseline))<1e-12
    assert np.max(abs(blend-q.baseline))<1e-12,'Need clipping decomposition if active'
    s=support[(support.context==context)&(support.farm==farm)&(support.day==day)].iloc[0];assert abs(float(et.mean())-s.prediction)<1e-10
    truth=float(q.y.mean());bias=float(q.baseline.mean()-truth)
    row=dict(context=context,farm=farm,day=day,seed=7,true_day=truth,final_prediction=float(q.baseline.mean()),final_bias=bias,ET_prediction=float(et.mean()),ET_weighted_bias=.48*(float(et.mean())-truth),LGB_weighted_bias=.24*(float(lp.mean())-truth),MLP_weighted_bias=.08*(float(mp.mean())-truth),PFN_weighted_bias=.2*(float(pp.mean())-truth),ET_fraction_of_signed_bias=.48*(float(et.mean())-truth)/bias)
    assert abs(sum(row[n+'_weighted_bias'] for n in ['ET','LGB','MLP','PFN'])-bias)<1e-12;out.append(row)
dest=H/'composition_v1.json'
with dest.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_SIGNED_BIAS_DECOMPOSITION',source_oof_sha=hashlib.sha256(P.read_bytes()).hexdigest(),rows=out,new_fit=0,clipping_active=False,PFN_learning_traced=False),f,ensure_ascii=False,indent=2)
print(json.dumps(out,indent=2))
