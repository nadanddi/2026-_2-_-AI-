from pathlib import Path
import sys,os,json,hashlib,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'));import env_extra_gpu
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd,torch
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
OUT=ROOT/'집/코덱스/local/ec_matched_inner_gpu_preview_20261003_v1'
CPU_OUT=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
CKPT=Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'
def basis(x):return np.column_stack([np.ones(len(x)),x,np.maximum(x-.6,0),np.maximum(x-1,0)])
def main():
 OUT.mkdir(parents=True,exist_ok=True);sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();p=OUT/'source_sha.txt'
 if p.exists():assert p.read_text()==sha
 else:p.write_text(sha)
 assert hashlib.sha256(CKPT.read_bytes()).hexdigest()=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 from tabpfn.architectures.tabpfn_v2 import TabPFNV2
 def cpu_seed_embeddings(self,x):
  gen=torch.Generator(device='cpu').manual_seed(self.seed);nc,sz=x.shape[2],x.shape[3]
  embs=torch.randn((nc,sz//4),device='cpu',dtype=x.dtype,generator=gen).to(x.device)
  return x+self.feature_positional_embedding_embeddings(embs)[None,None]
 TabPFNV2.add_column_embeddings=cpu_seed_embeddings
 lab,core,wv,folds,outer=S.loadec();idx=lab.set_index('row_id');full=[c for c in core.FULL if c!='day']+['season'];base=[c for c in core.BASE if c!='day']+['season'];coef=[];checks=0
 for v,k,tm,vm in folds:
  tr=lab[tm];va=lab[vm].reset_index(drop=True);old=dict(np.load(S.OUT/f'E_{v}_{k}_cpu.npz'));a=idx.reindex(old['inner_train_id']).reset_index();b=idx.reindex(old['row_id']).reset_index()
  assert set(a.row_id).isdisjoint(b.row_id) and set(a.row_id)|set(b.row_id)<=set(tr.row_id)
  banned={(f,int(d)+offset) for f,d in b[['farm','day']].drop_duplicates().itertuples(index=False,name=None) for offset in [-1,0,1]};assert not set(a[['farm','day']].itertuples(index=False,name=None))&banned
  a,b=S.seasonal(a,b,wv);cp=OUT/f'{v}_{k}_cpu.npz'
  if not cp.exists() and (CPU_OUT/cp.name).exists():
   with np.load(CPU_OUT/cp.name) as src:copied={key:src[key].copy() for key in src.files}
   assert np.array_equal(copied['row_id'],b.row_id) and np.array_equal(copied['inner_train_id'],a.row_id)
   np.savez(cp,**copied);print(v,k,'canonical CPU cache reused',flush=True)
  if not cp.exists():
   payload=dict(row_id=b.row_id.to_numpy(str),inner_train_id=a.row_id.to_numpy(str),lo=a.sub_ec.min(),hi=a.sub_ec.max())
   for seed in [7,101,2024]:
    pred=[];models=[(core.et(seed),full),(core.lg(seed,'tweedie'),base),(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed)),base)]
    with threadpool_limits(limits=2):
     for m,cols in models:
      if hasattr(m,'steps') and isinstance(m.steps[-1][1],core.ExtraTreesRegressor):m.steps[-1][1].n_jobs=2
      m.fit(a[cols],a.sub_ec);pred.append(m.predict(b[cols]))
    payload[f'r3_{seed}']=.6*pred[0]+.3*pred[1]+.1*pred[2]
    if v=='DIAG10' and k==0 and seed==7:
     repeat=[]
     with threadpool_limits(limits=2):
      for m,cols in models:
       fresh=clone(m);fresh.fit(a[cols],a.sub_ec);repeat.append(fresh.predict(b[cols]))
     diff=float(np.max(np.abs((.6*repeat[0]+.3*repeat[1]+.1*repeat[2])-payload[f'r3_{seed}'])));assert diff<1e-9;(H/'cpu_reproduction_v1.json').write_text(json.dumps(dict(status='PASS',maxdiff=diff)),encoding='utf-8')
   np.savez(cp,**payload);print(v,k,'CPU done',flush=True)
  z=dict(np.load(cp));assert np.array_equal(z['row_id'],b.row_id) and np.array_equal(z['inner_train_id'],a.row_id)
  pp=[]
  for seed in [1,2,3,4]:
   pc=OUT/f'{v}_{k}_pfn_{seed}.npz'
   if not pc.exists() and v=='DIAG10' and k==0:
    bg=dict(np.load(CPU_OUT/'gpu_cpu_rng_bundle_first.npz'));assert np.array_equal(bg['row_id'],b.row_id)
    ci=np.random.default_rng(seed).choice(len(a),min(2000,len(a)),replace=False)
    np.savez(pc,row_id=b.row_id.to_numpy(str),context_row_id=a.row_id.iloc[ci].to_numpy(str),prediction=bg['prediction_by_context'][seed-1]);print(v,k,'GPU bundle reused',seed,flush=True)
   if not pc.exists():
    ix=np.random.default_rng(seed).choice(len(a),min(2000,len(a)),replace=False)
    def fit():
     m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',model_path=str(CKPT),n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
     with threadpool_limits(limits=4):
      m.fit(a[full].to_numpy(np.float32)[ix],a.sub_ec.to_numpy(float)[ix]);pred=np.asarray(m.predict(b[full].to_numpy(np.float32)),float);small=np.asarray(m.predict(b.iloc[:8][full].to_numpy(np.float32)),float)
     assert np.array_equal(pred[:8],small);return pred
    pred=fit();assert np.isfinite(pred).all()
    if v=='DIAG10' and k==1 and seed==1:
     repeat=fit();diff=float(np.max(np.abs(repeat-pred)));assert diff<1e-5;(H/'pfn_reproduction_v1.json').write_text(json.dumps(dict(status='PASS',maxdiff=diff)),encoding='utf-8')
    np.savez(pc,row_id=b.row_id.to_numpy(str),context_row_id=a.row_id.iloc[ix].to_numpy(str),prediction=pred);print(v,k,'PFN',seed,'done',flush=True);gc.collect()
   zz=dict(np.load(pc));assert np.array_equal(zz['row_id'],b.row_id) and set(zz['context_row_id'])<=set(a.row_id);pp.append(zz['prediction'])
  bag=np.mean(pp,axis=0);yday=b.groupby(['farm','day']).sub_ec.mean()
  for seed in [7,101,2024]:
   baseline=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi']);ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy();cor=np.zeros(len(va))
   for h in range(24):
    x=b.loc[b.hour<=h,['farm','day']].assign(p=baseline[b.hour<=h]).groupby(['farm','day']).p.mean();X=basis(x.to_numpy());t=yday.reindex(x.index).to_numpy()-x.to_numpy();design=np.vstack([X,np.sqrt(10)*np.eye(4)]);target=np.r_[t,np.zeros(4)];m=lsq_linear(design,target,bounds=([-np.inf,-1,0,0],[np.inf,np.inf,np.inf,np.inf]),tol=1e-12,max_iter=300);assert m.success
    beta=m.x;grad=design.T@(design@beta-target);lower=np.array([-np.inf,-1,0,0]);active=np.isfinite(lower)&(beta-lower<1e-6);assert np.max(np.abs(grad[~active]))<1e-6 and (not active.any() or np.min(grad[active])>-1e-6)
    q=va.loc[va.hour<=h,['farm','day']].assign(p=ref[va.hour<=h]).groupby(['farm','day']).p.mean();use=va.hour==h;xp=q.reindex(pd.MultiIndex.from_frame(va.loc[use,['farm','day']])).to_numpy();cor[use]=.2*np.clip(basis(xp)@beta,-.3,.3);coef.append(dict(validator=v,fold=k,seed=seed,hour=h,beta=beta.tolist(),n_days=len(x)));checks+=1
   d=va[['row_id','farm','day','hour']].copy();d['y']=va.sub_ec;d['baseline']=ref;d['candidate']=np.clip(ref+cor,tr.sub_ec.min(),tr.sub_ec.max());d['correction']=cor;d['validator']=v;d['fold']=k;d['seed']=seed;d['clip_lo']=tr.sub_ec.min();d['clip_hi']=tr.sub_ec.max();d.to_csv(OUT/f'{v}_{k}_{seed}_pred.csv',index=False)
  print(v,k,'FOLD COMPLETE',flush=True)
 pd.concat([pd.read_csv(p,float_precision='round_trip') for p in OUT.glob('*_pred.csv')],ignore_index=True).to_csv(OUT/'oof.csv',index=False)
 (H/'fit_audit_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,coefficients=coef),ensure_ascii=False,indent=2),encoding='utf-8');print('GPU_PREVIEW_ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
