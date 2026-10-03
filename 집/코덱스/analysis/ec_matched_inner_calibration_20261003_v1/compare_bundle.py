from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
O=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1';g=dict(np.load(O/'gpu_cpu_rng_bundle_first.npz'));z=dict(np.load(O/'DIAG10_0_cpu.npz'));pp=[]
for seed in [1,2,3,4]:
 c=dict(np.load(O/f'DIAG10_0_pfn_{seed}.npz'));assert np.array_equal(c['row_id'],g['row_id']);assert set(c['context_row_id'])<=set(z['inner_train_id']);pp.append(c['prediction'])
cp=np.mean(pp,axis=0);gp=np.mean(g['prediction_by_context'],axis=0);diff=cp-gp;maxraw=max(abs(float(x)-float(y)) for x,y in zip(cp,gp));assert abs(maxraw-np.max(np.abs(diff)))<1e-12
lab,core,wv,folds,outer=S.loadec();v,k,tm,vm=folds[0];assert v=='DIAG10' and k==0;b=lab.set_index('row_id').reindex(z['row_id']).reset_index();va=lab[vm].reset_index(drop=True);yday=b.groupby(['farm','day']).sub_ec.mean();results=[]
def basis(x):return np.column_stack([np.ones(len(x)),x,np.maximum(x-.6,0),np.maximum(x-1,0)])
with threadpool_limits(limits=2):
 for seed in [7,101,2024]:
  cb=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*cp,b),z['lo'],z['hi']);gb=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*gp,b),z['lo'],z['hi']);ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy();cor=np.zeros(len(va))
  for h in range(24):
   x=b.loc[b.hour<=h,['farm','day']].assign(p=gb[b.hour<=h]).groupby(['farm','day']).p.mean();X=basis(x.to_numpy());target=yday.reindex(x.index).to_numpy()-x.to_numpy();m=lsq_linear(np.vstack([X,np.sqrt(10)*np.eye(4)]),np.r_[target,np.zeros(4)],bounds=([-np.inf,-1,0,0],[np.inf,np.inf,np.inf,np.inf]),tol=1e-12,max_iter=300);assert m.success
   q=va.loc[va.hour<=h,['farm','day']].assign(p=ref[va.hour<=h]).groupby(['farm','day']).p.mean();use=va.hour==h;xp=q.reindex(pd.MultiIndex.from_frame(va.loc[use,['farm','day']])).to_numpy();cor[use]=.2*np.clip(basis(xp)@m.x,-.3,.3)
  gpu=np.clip(ref+cor,lab[tm].sub_ec.min(),lab[tm].sub_ec.max());cpu=pd.read_csv(O/f'DIAG10_0_{seed}_pred.csv',float_precision='round_trip');assert np.array_equal(cpu.row_id,va.row_id)
  d=gpu-cpu.candidate.to_numpy();md=max(abs(float(x)) for x in d);assert abs(md-np.max(np.abs(d)))<1e-12;results.append(dict(seed=seed,rows=len(va),inner_baseline_maxdiff=float(np.max(np.abs(cb-gb))),candidate_maxdiff=md,candidate_rmsdiff=math.sqrt(math.fsum(float(x)**2 for x in d)/len(d))))
passed=all(x['inner_baseline_maxdiff']<1e-5 and x['candidate_maxdiff']<1e-5 for x in results)
(H/'bundle_equivalence_v1.json').write_text(json.dumps(dict(status='COMPUTATION_MATCH' if passed else 'COMPUTATION_DIFFERENCE',raw_PFN_bag_maxdiff=maxraw,results=results,all_under_1e_5=passed,limitations=['first fold only','individual raw PFN equivalence earlier failed','no model performance judgment or CPU replacement','full CPU experiment still running']),ensure_ascii=False,indent=2),encoding='utf-8');print('PASSED',passed);print(json.dumps(results,indent=2))
