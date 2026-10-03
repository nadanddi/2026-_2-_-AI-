from pathlib import Path
import sys,os,json,gc,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'));import env_extra_gpu
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.architectures.tabpfn_v2 import TabPFNV2
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_gpu_fixed_batch_audit_20261003_v1'
CPU=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
CKPT=Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'
def cpu_seed_embeddings(self,x):
 generator=torch.Generator(device='cpu').manual_seed(self.seed);nc,sz=x.shape[2],x.shape[3]
 embs=torch.randn((nc,sz//4),device='cpu',dtype=x.dtype,generator=generator).to(x.device)
 return x+self.feature_positional_embedding_embeddings(embs)[None,None]
def fixed_predict(m,x,anchor):
 pieces=[]
 for i in range(0,len(x),2048):
  block=x[i:i+2048];pad=np.repeat(anchor[None],2048,axis=0);pad[:len(block)]=block
  p=np.asarray(m.predict(pad),float);assert len(p)==2048
  pieces.append(p[:len(block)])
 return np.concatenate(pieces)
def main():
 OUT.mkdir(parents=True,exist_ok=True);assert torch.cuda.is_available()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 assert hashlib.sha256(CKPT.read_bytes()).hexdigest()=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
 TabPFNV2.add_column_embeddings=cpu_seed_embeddings
 lab,core,wv,folds,outer=S.loadec();z=dict(np.load(CPU/'DIAG10_1_cpu.npz'));a=lab.set_index('row_id').reindex(z['inner_train_id']).reset_index();b=lab.set_index('row_id').reindex(z['row_id']).reset_index();a,b=S.seasonal(a,b,wv)
 cols=[c for c in core.FULL if c!='day']+['season'];ix=np.random.default_rng(1).choice(len(a),min(2000,len(a)),replace=False);train=a[cols].to_numpy(np.float32)[ix];query=b[cols].to_numpy(np.float32);anchor=train[0].copy();runs=[];checks=[]
 for n in [1,2]:
  m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',model_path=str(CKPT),n_estimators=4,random_state=1,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
  with threadpool_limits(limits=2):
   m.fit(train,a.sub_ec.to_numpy(float)[ix]);p=fixed_predict(m,query,anchor);assert np.isfinite(p).all()
   for count in [8,17]:
    small=fixed_predict(m,query[:count],anchor);diff=float(np.max(np.abs(p[:count]-small)));checks.append(dict(fit=n,kind=f'prefix{count}',maxdiff=diff,bit_equal=bool(np.array_equal(p[:count],small))))
   changed=query.copy();changed[8:]=changed[8:]*17+1000;diffp=fixed_predict(m,changed,anchor);checks.append(dict(fit=n,kind='other_query_changed_prefix8',maxdiff=float(np.max(np.abs(p[:8]-diffp[:8]))),bit_equal=bool(np.array_equal(p[:8],diffp[:8]))))
   perm=np.random.default_rng(42).permutation(len(query));restored=fixed_predict(m,query[perm],anchor)[np.argsort(perm)];checks.append(dict(fit=n,kind='row_order',maxdiff=float(np.max(np.abs(p-restored))),bit_equal=bool(np.array_equal(p,restored))))
  runs.append(p);del m;gc.collect();torch.cuda.empty_cache();print('FIT',n,'COMPLETE',flush=True)
 cpu=dict(np.load(CPU/'DIAG10_1_pfn_1.npz'));assert np.array_equal(cpu['row_id'],b.row_id)
 raw=float(np.max(np.abs(runs[0]-cpu['prediction'])));repeat=float(np.max(np.abs(runs[0]-runs[1])))
 bit=bool(np.array_equal(runs[0],runs[1]) and all(c['bit_equal'] for c in checks))
 np.savez(OUT/'fold1_context1_fixed.npz',row_id=b.row_id.to_numpy(str),context_row_id=a.row_id.iloc[ix].to_numpy(str),prediction=runs[0])
 result=dict(status='AUDIT_COMPLETE',query_invariance_pass=bit,cpu_raw_equivalence_pass=bool(raw<=1e-5),repeat_maxdiff=repeat,cpu_raw_maxdiff=raw,batch_size=2048,query_rows=len(query),checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),warning='Not a score or adopted model; CPU experiment retained; padding is a fixed training input only')
 (H/'result_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
