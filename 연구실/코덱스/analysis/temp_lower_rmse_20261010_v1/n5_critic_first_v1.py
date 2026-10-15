"""Independent scalar prefix and frozen-array verification, no model fits."""
from pathlib import Path
import sys,json,hashlib,math
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np,pandas as pd
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
reg=json.loads((H/'n5_registration_v2.json').read_text(encoding='utf-8'))
parent=json.loads((H/'n4_registration_v2.json').read_text(encoding='utf-8'))
prep=json.loads((H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
assert reg['source_sha']==sha(H/'n5_run_v2.py')
assert reg['parent_recipe_sha']==canon(parent) and reg['parent_registration_sha']==sha(H/'n4_registration_v2.json')
assert reg['parent_source_receipt_sha']==sha(H/'n4_source_receipt_v1.json')
assert reg['folds']==parent['folds'] and reg['data_sha']==parent['data_sha']==prep['data_sha']
data=Path(prep['temp_code']).parent/'data'
for name,digest in reg['data_sha'].items():assert sha(data/name)==digest
audit=json.loads((H/'n5_audit_v2.json').read_text(encoding='utf-8'));assert audit['recipe_sha']==canon(reg)
raw=pd.concat([pd.read_csv(data/n,usecols=['row_id','in_temp']) for n in ['train_X.csv','test_X.csv']])
raw=raw[raw.row_id.str[:3].isin(['F13','F47'])]
assert raw.row_id.is_unique and np.isfinite(raw.in_temp.dropna()).all()
states={};previous={};packets={}
for rid,t in sorted(zip(raw.row_id,raw.in_temp)):
 key=(rid[:3],int(rid[4:7]));hr=int(rid[8:10]);packets.setdefault(key,[]).append(hr)
 old=previous.get(key,float('nan'))
 current=float(t)
 minimum=old if math.isnan(current) else current if math.isnan(old) else min(old,current)
 q=1. if math.isnan(minimum) else min(1.,max(0.,(minimum-8.)/2.))
 g=1. if math.isnan(current) else min(1.,max(0.,(current-8.)/2.))
 if key in previous:assert q<=states[last_ids[key]][1]
 else:
  if 'last_ids' not in globals():last_ids={}
 assert q<=g
 states[rid]=(minimum,q,g);previous[key]=minimum;last_ids[key]=rid
assert len(packets)==460 and all(hours==list(range(24)) for hours in packets.values())
fd=reg['folds'][0];name=fd['set']+'_'+str(fd['fold'])
O=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1'
pf=O/'n4_v1'/(name+'.npz');nf=O/'n5_v1'/(name+'.npz')
pm=json.loads(pf.with_suffix('.json').read_text(encoding='utf-8'));nm=json.loads(nf.with_suffix('.json').read_text(encoding='utf-8'))
assert sha(pf)==reg['parents'][name]['npz_sha']==pm['npz_sha']==nm['parent_npz_sha']
assert sha(pf.with_suffix('.json'))==reg['parents'][name]['receipt_sha']
assert sha(nf)==nm['npz_sha'] and nm['recipe_sha']==canon(reg)
z=np.load(pf,allow_pickle=True);v=np.load(nf,allow_pickle=True);ids=v['row_id'].tolist()
assert ids==z['row_id'].tolist()==fd['query_ids']==nm['query_ids']==pm['query_ids']
assert len(ids)==960 and len(set(ids))==960 and not set(ids)&set(fd['train_ids'])
mn=np.array([states[i][0] for i in ids]);q=np.array([states[i][1] for i in ids]);g=np.array([states[i][2] for i in ids])
assert np.array_equal(mn,v['prefix_min'],equal_nan=True)
assert np.array_equal(q,v['q']) and np.array_equal(g,v['g']) and np.array_equal(g,z['gate'])
assert np.array_equal(v['BASELINE'],z['BASELINE'])
assert np.array_equal(v['N5'][q==g],z['N4'][q==g])
assert np.array_equal(v['N5'][q==0],z['BASELINE'][q==0])
assert np.array_equal(v['N5'][g==0],z['BASELINE'][g==0])
gaps=[]
for b,c,irr,p,qq,gg,out in zip(z['BASE'],z['CODEX'],z['IRR'],z['PFN'],q,g,v['N5']):
 calc=math.fsum([.4*float(b),(.6-.4*float(gg))*(float(c)+float(qq)*(float(irr)-float(c))),.4*float(gg)*float(p)])
 gaps.append(abs(calc-float(out)))
assert max(gaps)<1e-13 and np.isfinite(v['N5']).all()
truth=pd.read_csv(data/'train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').loc[ids,'sub_temp'].to_numpy()
rm=lambda pred:math.sqrt(math.fsum((float(p)-float(t))**2 for p,t in zip(pred,truth))/len(ids))
newp=rm(v['N5']);oldp=rm(v['BASELINE'])
d=v['N5']-v['BASELINE'];cross=2*math.fsum((float(p)-float(t))*float(c) for p,t,c in zip(v['BASELINE'],truth,d))/len(ids);dc=math.fsum(float(c)**2 for c in d)/len(ids)
assert abs(cross+dc-(newp*newp-oldp*oldp))<1e-12
receipt={'status':'PASS_FIRST_FOLD_ONLY','query_rows':len(ids),'prefix_min_scalar_exact':True,'q_g_exact':True,'same_day_monotonic':True,'all460days_complete24':True,'parent_baseline_exact':True,'q_equal_g_N4_exact':True,'q0_rows':int((q==0).sum()),'g0_rows':int((g==0).sum()),'q_reduced_rows':int((q<g).sum()),'max_formula_gap':max(gaps),'base_rmse':oldp,'candidate_rmse':newp,'delta_mse':newp*newp-oldp*oldp,'cross_2E':cross,'correction_mse':dc,'new_fit':0,'ec_numeric_labels_read':False,'full_prediction_causality_test':False}
with (H/'n5_critic_first_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
