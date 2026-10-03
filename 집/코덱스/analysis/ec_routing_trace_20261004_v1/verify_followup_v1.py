from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
H=Path(__file__).parent;R={'scope':'public cached predictions plus saved routing; no fit/EL1/test/raw labels'}
src=ROOT/'집/클로드/research/local/ec3_TF1_all.csv'
d=pd.concat([c[c.validator!='EL1'] for c in pd.read_csv(src,chunksize=2048)],ignore_index=True)
assert not d.duplicated(['validator','validation_fold','row_id']).any()
cells=[]
for v,g in d.groupby('validator'):
 for s in [7,101,2024]:
  a=g[f'r3s_{s}'].to_numpy()-g.sub_ec.to_numpy();b=g[f'tf_{s}'].to_numpy()-g.sub_ec.to_numpy()
  ra,rb=np.sqrt(np.mean(a*a)),np.sqrt(np.mean(b*b))
  assert abs(ra-math.sqrt(math.fsum(float(z)**2 for z in a)/len(a)))<1e-12
  assert abs(rb-math.sqrt(math.fsum(float(z)**2 for z in b)/len(b)))<1e-12
  cells.append(dict(validator=v,seed=s,n=len(g),baseline=float(ra),candidate=float(rb),change_pct=float(100*(rb/ra-1))))
g=d[d.validator=='DIAG10'].copy();ps=[];rng=np.random.default_rng(20261004)
for s in [7,101,2024]:
 dd=(g[f'tf_{s}']-g.sub_ec)**2-(g[f'r3s_{s}']-g.sub_ec)**2
 cl=dd.groupby(g.farm+'_'+(g.day//5).astype(str)).agg(['sum','count']);ix=rng.integers(0,len(cl),(20000,len(cl)))
 ps.append(float((cl['sum'].to_numpy()[ix].sum(1)/cl['count'].to_numpy()[ix].sum(1)>=0).mean()))
R['TF1']={'cells':cells,'p_worse':ps,'source_sha':hashlib.sha256(src.read_bytes()).hexdigest(),'decision':'REJECT'}
q=pd.read_csv(H/'query_prefix_sources_v1.csv');s=pd.read_csv(H/'top_source_days_v1.csv')
z=q[(q.farm=='F47')&(q.day==132)&(q.hour==0)].iloc[0]
manual=z.high_training_day_mass*z.selected_high_row_mean+(1-z.high_training_day_mass)*z.selected_other_row_mean
assert abs(manual-z.et_shrunk)<1e-12
assert len(q)==28*4 and len(s)==28*4*5
R['hard_case']={k:float(z[k]) for k in ['actual_day_mean','actual_current','actual_v2_current','et_shrunk','high_training_day_mass','selected_high_row_mean','selected_other_row_mean','effective_source_days','train_day_max']}
R['hard_case']['component_identity_maxdiff']=abs(manual-z.et_shrunk)
# Independently repeat actual-baseline directional signs without rewriting critic files.
hm=pd.concat([c[c.validator!='EL1'] for c in pd.read_csv(ROOT/'집/클로드/research/local/ec3_HM1_all.csv',chunksize=2048)],ignore_index=True)
out=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv',float_precision='round_trip')
ds=[]
for seed in [7,101,2024]:
 p=out[out.seed==seed][['row_id','validator','validation_fold','sub_ec','season_v2']].rename(columns={'sub_ec':'actual_y'})
 h=hm.merge(p,on=['row_id','validator','validation_fold'],validate='one_to_one');assert len(h)==len(hm) and np.max(abs(h.actual_y-h.sub_ec))<1e-12
 for v,g in h.groupby('validator'):
  direction=np.where(g[f'flag_{seed}'],g[f'sp_{seed}']-g[f'r3s_{seed}'],0)
  derivative=2*math.fsum(float(e)*float(x) for e,x in zip(g.season_v2-g.sub_ec,direction))/len(g)
  assert abs(derivative-2*np.mean((g.season_v2-g.sub_ec)*direction))<1e-12
  ds.append(dict(validator=v,seed=seed,derivative_at_zero=derivative))
R['HM1_actual_baseline']={'cells':ds,'favourable_DIAG_A_B':sum(x['derivative_at_zero']<0 for x in ds if x['validator'] in ['DIAG10','A','B']),'denominator':9}
# Independent algebra checks; not EC observations.
toy={'original_mean':.9*0+.1*2,'weighted_mean':(.1*3*2)/(.9+.1*3)}
toy['original_mse']=.9*toy['original_mean']**2+.1*(2-toy['original_mean'])**2
toy['weighted_optimum_original_mse']=.9*toy['weighted_mean']**2+.1*(2-toy['weighted_mean'])**2
assert abs(toy['original_mse']-.36)<1e-12 and abs(toy['weighted_optimum_original_mse']-.45)<1e-12
R['loss_toy']=toy
dest=H/'followup_verification_v1.json';assert not dest.exists();dest.write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'TF1_p':ps,'TF1_pct':cells,'hard_case':R['hard_case'],'HM1_favourable':R['HM1_actual_baseline']['favourable_DIAG_A_B']},ensure_ascii=False,indent=2))
