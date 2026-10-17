"""Independent saved-output verification; no fits or EC numeric labels."""
from pathlib import Path
import sys,json,hashlib,math
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'temp_repair_20261009_v1'))
import runtime_v1 as r
np=r.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((H/'n4_registration_v2.json').read_text(encoding='utf-8'))
src=json.loads((H/'n4_source_receipt_v1.json').read_text(encoding='utf-8'))
result=json.loads((H/'n4_results_v2.json').read_text(encoding='utf-8'))
groups=json.loads((H/'n4_groups_v2.json').read_text(encoding='utf-8'))
rs=hashlib.sha256(json.dumps(reg,sort_keys=True).encode()).hexdigest()
assert reg['source_sha']==src['script_sha']==sha(H/'n4_run_v2.py')
assert reg['runtime_sha']==src['runtime_sha']==sha(H.parent/'temp_repair_20261009_v1/runtime_v1.py')
assert src['preparation_sha']==sha(H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json')
assert src['data_sha']==reg['data_sha']==r.P['data_sha']
assert src['package_files']==r.P['temp_package_files']
for p,v in src['package_files'].items():assert sha(r.CODE.parent/p)==v
for p,v in src['data_sha'].items():assert sha(r.DATA/p)==v
assert sha(Path(r.P['cache'])/'tabpfn-v2-regressor.ckpt')==src['checkpoint_sha']==reg['checkpoint_sha']
lab=r.RAW[1].set_index('row_id',drop=False);raw=r.RAW[0].set_index('row_id')
ans={s:{'ids':[],'a':[],'b':[],'g':[],'air':[]} for s in ['DIAG10','EXT10','EXT12']}
maxgap=0.;checks=0
def close(a,b):
 global maxgap,checks
 gap=abs(a-b);maxgap=max(maxgap,gap);checks+=1;assert gap<1e-12,(a,b)
O=r.ROOT/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n4_v1'
for fd in reg['folds']:
 fn=O/(fd['set']+'_'+str(fd['fold'])+'.npz');m=json.loads(fn.with_suffix('.json').read_text(encoding='utf-8'))
 assert m['recipe_sha']==rs and m['npz_sha']==sha(fn)
 assert m['train_ids']==fd['train_ids'] and m['query_ids']==fd['query_ids']
 z=np.load(fn,allow_pickle=True);ids=z['row_id'].tolist()
 assert ids==fd['query_ids'] and len(ids)==len(set(ids)) and not set(ids)&set(fd['train_ids'])
 parse=lambda i:(i[:3],int(i[4:7]),int(i[8:10]))
 td={(f,d) for f,d,h in map(parse,fd['train_ids'])};qd={(f,d) for f,d,h in map(parse,ids)}
 assert all(f!=g or abs(d-e)>1 for f,d in td for g,e in qd)
 for seq in [ids,fd['train_ids']]:
  days={}
  for i in seq:
   f,d,h=parse(i);days.setdefault((f,d),[]).append(h)
  assert all(sorted(hs)==list(range(24)) for hs in days.values())
 tr=lab.loc[fd['train_ids']].reset_index(drop=True);va=lab.loc[ids].drop(columns=['sub_temp','sub_ec']).reset_index(drop=True)
 w=r.weights(tr);assert {str(v):int((w==v).sum()) for v in np.unique(w)}==m['weights']
 _,_,notes=r.season_frames(tr,va);assert notes==m['season']
 idx=np.random.default_rng(49).choice(len(tr),size=min(2000,len(tr)),replace=False,p=w/w.sum())
 assert np.array_equal(tr.row_id.to_numpy()[idx],z['context_ids'])
 assert len(set(z['context_ids']))==2000 and set(z['context_ids'])<=set(fd['train_ids'])
 air=raw.loc[ids,'in_temp'].to_numpy();g=np.array([1. if np.isnan(t) else min(1.,max(0.,(float(t)-8.)/2.)) for t in air])
 assert np.array_equal(g,z['gate']) and m['seed_noop_max']==0.
 assert all(np.isfinite(z[k]).all() for k in ['BASE','CODEX','IRR','PFN','BASELINE','N4','gate'])
 for b,c,irr,p,t,old,new in zip(z['BASE'],z['CODEX'],z['IRR'],z['PFN'],g,z['BASELINE'],z['N4']):
  aa=math.fsum([.4*float(b),(.6-.4*float(t))*float(c),.4*float(t)*float(p)])
  bb=math.fsum([aa,(.6-.4*float(t))*float(t)*(float(irr)-float(c))]);close(aa,float(old));close(bb,float(new))
 assert np.array_equal(z['BASELINE'][g==0],z['N4'][g==0])
 for k,v in [('ids',ids),('a',z['BASELINE']),('b',z['N4']),('g',g),('air',air)]:ans[fd['set']][k].extend(v)
print('ALL12_SOURCE_CONTEXT_WEIGHTS_SEASON_PASS',flush=True)
metrics={}
for s,d in ans.items():
 ids=d['ids'];assert len(ids)==len(set(ids));y=lab.loc[ids,'sub_temp'].to_numpy();a=np.array(d['a']);b=np.array(d['b']);air=np.array(d['air'])
 ea=[(float(p)-float(t))**2 for p,t in zip(a,y)];eb=[(float(p)-float(t))**2 for p,t in zip(b,y)]
 def metric(ix):
  n=len(ix);ma=math.fsum(ea[i] for i in ix)/n;mb=math.fsum(eb[i] for i in ix)/n;ra=math.sqrt(ma);rb=math.sqrt(mb)
  return {'n':n,'base_rmse':ra,'candidate_rmse':rb,'pct':100*(rb/ra-1),'delta_mse':mb-ma}
 whole=metric(list(range(len(ids))))
 for k in ['base_rmse','candidate_rmse','pct']:close(whole[k],result['results'][s][k])
 for k,v in whole.items():close(v,groups[s]['overall'][k])
 for f in ['F13','F47']:
  for late in [False,True]:
   ix=[i for i,row in enumerate(ids) if row[:3]==f and (int(row[4:7])>=179)==late]
   if ix:
    got=metric(ix);target=groups[s]['farm_segment'][str((f,'second' if late else 'first'))]
    for k,v in got.items():close(v,target[k])
 for label,pred in [('in_temp_le8',lambda t:not np.isnan(t) and t<=8),('in_temp_8to10',lambda t:not np.isnan(t) and 8<t<10),('in_temp_ge10',lambda t:not np.isnan(t) and t>=10),('in_temp_NaN',lambda t:np.isnan(t))]:
  ix=[i for i,t in enumerate(air) if pred(t)]
  if ix:
   for k,v in metric(ix).items():close(v,groups[s]['gate'][label][k])
 correction=b-a;cross=2*math.fsum((float(p)-float(t))*float(q) for p,t,q in zip(a,y,correction))/len(ids)
 cmse=math.fsum(float(q)**2 for q in correction)/len(ids)
 close(cross,groups[s]['error_delta_cross_2E']);close(cmse,groups[s]['correction_mse']);close(cross+cmse,whole['delta_mse'])
 if s=='DIAG10':
  buckets={}
  for i,row in enumerate(ids):buckets.setdefault(row[:3]+'_'+str(int(row[4:7])//5),[]).append(i)
  keys=sorted(buckets);sa=np.array([math.fsum(ea[i] for i in buckets[k]) for k in keys]);sb=np.array([math.fsum(eb[i] for i in buckets[k]) for k in keys]);sz=np.array([len(buckets[k]) for k in keys]);rng=np.random.default_rng(reg['bootstrap_seed']);draws=[]
  for _ in range(20):
   ix=rng.integers(len(keys),size=(1000,len(keys)));den=sz[ix].sum(1);draws.extend((np.sqrt(sb[ix].sum(1)/den)-np.sqrt(sa[ix].sum(1)/den)).tolist())
  pv=float(np.mean(np.array(draws)>=0));ci=np.quantile(draws,[reg['alpha'],1-reg['alpha']]).tolist();close(pv,result['results'][s]['p_worse'])
  for x,zv in zip(ci,result['results'][s]['ci']):close(x,zv)
  whole.update(p_worse=pv,ci=ci)
 metrics[s]=whole
passed=all(v['candidate_rmse']<v['base_rmse'] for v in metrics.values()) and metrics['DIAG10']['p_worse']<reg['alpha'] and metrics['DIAG10']['ci'][1]<0
assert passed==result['pilot_pass']==False and result['adoption']==False
out={'status':'PASS_NUMERICS_PILOT_FAIL','checks':checks,'max_abs_gap':maxgap,'all12_fold_context_weights_season_source_pass':True,'metrics':metrics,'new_fit':0,'ec_numeric_labels_read':False,'full_prediction_causality_test':False}
with (H/'n4_critic_all_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
