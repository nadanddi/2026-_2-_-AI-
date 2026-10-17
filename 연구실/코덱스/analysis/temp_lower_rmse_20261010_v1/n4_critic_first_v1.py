"""Independent first-fold audit, no model fit or EC-label parsing."""
from pathlib import Path
import sys,json,hashlib,math
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'temp_repair_20261009_v1'))
import runtime_v1 as r
np=r.np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((H/'n4_registration_v2.json').read_text(encoding='utf-8'))
src=json.loads((H/'n4_source_receipt_v1.json').read_text(encoding='utf-8'))
rs=hashlib.sha256(json.dumps(reg,sort_keys=True).encode()).hexdigest()
assert reg['source_sha']==src['script_sha']==sha(H/'n4_run_v2.py')
assert reg['runtime_sha']==src['runtime_sha']==sha(H.parent/'temp_repair_20261009_v1/runtime_v1.py')
assert src['preparation_sha']==sha(H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json')
assert src['data_sha']==reg['data_sha']==r.P['data_sha']
assert src['package_files']==r.P['temp_package_files']
for p,v in src['package_files'].items():assert sha(r.CODE.parent/p)==v
for p,v in src['data_sha'].items():assert sha(r.DATA/p)==v
assert sha(Path(r.P['cache'])/'tabpfn-v2-regressor.ckpt')==src['checkpoint_sha']==reg['checkpoint_sha']
fd=reg['folds'][0]
fn=r.ROOT/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n4_v1/DIAG10_0.npz'
m=json.loads(fn.with_suffix('.json').read_text(encoding='utf-8'))
assert m['recipe_sha']==rs and m['npz_sha']==sha(fn)
assert m['train_ids']==fd['train_ids'] and m['query_ids']==fd['query_ids']
z=np.load(fn,allow_pickle=True);ids=z['row_id'].tolist()
assert ids==fd['query_ids'] and len(ids)==len(set(ids))
assert not set(ids)&set(fd['train_ids'])
parse=lambda i:(i[:3],int(i[4:7]),int(i[8:10]))
trdays={(f,d) for f,d,h in map(parse,fd['train_ids'])}
qdays={(f,d) for f,d,h in map(parse,ids)}
assert all(f!=g or abs(d-e)>1 for f,d in trdays for g,e in qdays)
for seq in [ids,fd['train_ids']]:
 days={}
 for i in seq:
  f,d,h=parse(i);days.setdefault((f,d),[]).append(h)
 assert all(sorted(hs)==list(range(24)) for hs in days.values())
lab=r.RAW[1].set_index('row_id',drop=False)
tr=lab.loc[fd['train_ids']].reset_index(drop=True)
va=lab.loc[ids].drop(columns=['sub_temp','sub_ec']).reset_index(drop=True)
w=r.weights(tr)
wd={str(v):int((w==v).sum()) for v in np.unique(w)}
assert wd==m['weights'] and len(w)==len(tr)
_,_,notes=r.season_frames(tr,va)
assert notes==m['season']
idx=np.random.default_rng(49).choice(len(tr),size=min(2000,len(tr)),replace=False,p=w/w.sum())
expected=tr.row_id.to_numpy()[idx]
assert np.array_equal(expected,z['context_ids'])
assert len(set(z['context_ids']))==2000 and set(z['context_ids'])<=set(fd['train_ids'])
air=r.RAW[0].set_index('row_id').loc[ids,'in_temp'].to_numpy()
g=np.array([1. if np.isnan(t) else min(1.,max(0.,(float(t)-8.)/2.)) for t in air])
assert np.array_equal(g,z['gate']) and m['seed_noop_max']==0.
truth=lab.loc[ids,'sub_temp'].to_numpy()
baseline=[];candidate=[]
for b,c,irr,p,t in zip(z['BASE'],z['CODEX'],z['IRR'],z['PFN'],g):
 baseline.append(math.fsum([.4*float(b),(.6-.4*float(t))*float(c),.4*float(t)*float(p)]))
 candidate.append(math.fsum([baseline[-1],(.6-.4*float(t))*float(t)*(float(irr)-float(c))]))
gap=max(max(abs(a-b) for a,b in zip(baseline,z['BASELINE'])),max(abs(a-b) for a,b in zip(candidate,z['N4'])))
assert gap<1e-13 and np.array_equal(z['BASELINE'][g==0],z['N4'][g==0])
assert all(np.isfinite(z[k]).all() for k in ['BASE','CODEX','IRR','PFN','BASELINE','N4','gate'])
rm=lambda a:math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(a,truth))/len(ids))
out={'status':'PASS_FIRST_FOLD_ONLY','source_package_files':len(src['package_files']),'train_rows':len(tr),'query_rows':len(ids),'context_rows':len(expected),'weighted_context_order_exact':True,'weights_exact':True,'season_exact':True,'buffer_24h_id_pass':True,'max_formula_gap':gap,'base_rmse':rm(z['BASELINE']),'candidate_rmse':rm(z['N4']),'gate0_rows':int((g==0).sum()),'no_op_max_reported':m['seed_noop_max'],'new_fit':0,'ec_numeric_labels_read':False,'full_prediction_causality_test':False}
with (H/'n4_critic_first_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
