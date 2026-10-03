from pathlib import Path
import sys,os,json,hashlib,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1';CKPT=OUT/'weights/tabpfn-v3.5-20260909.safetensors'
# No automatic account flow, license acceptance, or download from this runner.
if not CKPT.is_file():raise SystemExit('MISSING_LICENSED_WEIGHTS: obtain license consent and official checkpoint before running')
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'));import env_extra_gpu
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd,torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits
def predict(m,x,anchor):
 pieces=[]
 for i in range(0,len(x),2048):
  block=x[i:i+2048];padded=np.repeat(anchor[None],2048,axis=0);padded[:len(block)]=block;pieces.append(np.asarray(m.predict(padded),float)[:len(block)])
 return np.concatenate(pieces)
def main():
 assert torch.cuda.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 provenance=dict(source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint_sha=hashlib.sha256(CKPT.read_bytes()).hexdigest(),model_version='V3_5')
 prov=OUT/'provenance.json'
 if prov.exists():assert json.loads(prov.read_text())==provenance
 else:prov.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
 lab,core,wv,folds,outer=S.loadec();cols=[c for c in core.FULL if c!='day']+['season'];outputs=[]
 for v,k,tm,vm in folds:
  tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True);old=dict(np.load(OUT/f'{v}_{k}_baseline.npz'));assert np.array_equal(old['row_id'],va.row_id);pp=[]
  for c in [1,2,3,4]:
   cache=OUT/f'{v}_{k}_new_pfn_{c}.npz'
   if not cache.exists():
    context=tr.set_index('row_id').loc[old[f'context_{c}']];tx=context[cols].to_numpy(np.float32);ty=context.sub_ec.to_numpy(float);qx=va[cols].to_numpy(np.float32);anchor=tx[0].copy()
    def fit():
     model=TabPFNRegressor.create_default_for_version(ModelVersion.V3_5,model_path=str(CKPT),device='cuda',n_estimators=4,random_state=c,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
     with threadpool_limits(limits=2):
      model.fit(tx,ty);p=predict(model,qx,anchor);small=predict(model,qx[:8],anchor);assert np.array_equal(p[:8],small)
      if v=='DIAG10' and k==0 and c==1:
       altered=qx.copy();altered[8:]=altered[8:]*17+1000;assert np.array_equal(p[:8],predict(model,altered,anchor)[:8]);perm=np.random.default_rng(42).permutation(len(qx));assert np.array_equal(p,predict(model,qx[perm],anchor)[np.argsort(perm)])
     del model;gc.collect();torch.cuda.empty_cache();return p
    p=fit();assert np.isfinite(p).all()
    if v=='DIAG10' and k==0 and c==1:
     repeat=fit();assert np.array_equal(p,repeat);(H/'first_fold_verification_v1.json').write_text(json.dumps(dict(status='PASS',repeat_maxdiff=float(np.max(np.abs(p-repeat))),scope='new model first-fold context1; batch/order/other-query inference invariance'),indent=2),encoding='utf-8')
    np.savez(cache,row_id=va.row_id.to_numpy(str),context_row_id=context.index.to_numpy(str),prediction=p);print(v,k,'PFN',c,'done',flush=True)
   z=dict(np.load(cache));assert np.array_equal(z['row_id'],va.row_id) and np.array_equal(z['context_row_id'],old[f'context_{c}']);pp.append(z['prediction'])
  bag=np.mean(pp,axis=0)
  for seed in [7,101,2024]:
   raw=.8*old[f'r3_{seed}']+.2*bag;smoothed=core.shrink(raw,va);d=va[['row_id','farm','day','hour']].copy();d['y']=va.sub_ec;d['baseline']=old[f'baseline_{seed}'];d['candidate']=np.clip(smoothed,old['lo'],old['hi']);d['correction']=smoothed-d.baseline;d['new_pfn_raw']=bag;d['old_pfn_raw']=old['old_pfn_raw'];d['r3_raw']=old[f'r3_{seed}'];d['validator']=v;d['fold']=k;d['seed']=seed;d['clip_lo']=old['lo'];d['clip_hi']=old['hi'];d.to_csv(OUT/f'{v}_{k}_{seed}_pred.csv',index=False);outputs.append(d)
  print(v,k,'FOLD COMPLETE',flush=True)
 pd.concat(outputs,ignore_index=True).to_csv(OUT/'oof.csv',index=False);print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
