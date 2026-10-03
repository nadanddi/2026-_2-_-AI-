from prepare import *
import os,time,gc
for n in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[n]='1'
sys.path.insert(0,str(R/'.analysis-tools/extra'));import env_extra,torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
def main():
 torch.set_num_threads(4);world=joblib.load(O/'world.joblib');start=time.time();root=O/'pfn_ec_cpu';root.mkdir(exist_ok=True);core=S.loadcore();groupmap=json.loads((H/'ec_v3/EC_groups.json').read_text(encoding='utf-8'));effects=pd.read_csv(H/'effects_v2.csv');effects=effects[(effects.target=='EC')&(effects.seed==7)&(effects['rank']==1)];cols=[c for c in core.FULL if c!='day']+['season'];ckpt=Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'
 for fold in [3,6]:
  lab=world['elab'];_,_,tm,vm=next(z for z in world['efolds'] if z[0]=='DIAG10' and z[1]==fold);tr,va=S.seasonal(lab[tm],lab[vm],world['wv']);f,day=('F13',177) if fold==3 else ('F47',139);q=va[(va.farm==f)&(va.day==day)].sort_values('hour');specs=[dict(farm=f,day=day,group='ORIGINAL',donor_day=-1)];query=[q[cols].to_numpy(np.float32)]
  for e in effects[(effects.farm==f)&(effects.day==day)].itertuples():
   donor=tr[(tr.farm==f)&(tr.day==e.donor_day)].sort_values('hour');change=q.copy();cs=groupmap[e.group];change[cs]=donor[cs].to_numpy();query.append(change[cols].to_numpy(np.float32));specs.append(dict(farm=f,day=day,group=e.group,donor_day=int(e.donor_day)))
  query=np.concatenate(query)
  for seed in range(1,5):
   dest=root/f'PFN_EC_{fold}_{seed}.npz'
   if dest.exists():continue
   z=np.load(R/f'집/코덱스/local/ec_dc4_integration_20261002_v1/DIAG10_{fold}_pfn_{seed}.npz');context=z['context_row_id'].astype(str);idx=pd.Index(tr.row_id).get_indexer(context);assert (idx>=0).all()
   model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,model_path=str(ckpt),device='cpu',n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
   with S.threadpool_limits(limits=4):model.fit(tr[cols].to_numpy(np.float32)[idx],tr.sub_ec.to_numpy()[idx]);pred=np.asarray(model.predict(query))
   mp=dict(zip(z['row_id'].astype(str),z['raw_pfn']));delta=float(np.max(np.abs(pred[:24]-[mp[i] for i in q.row_id])));assert delta<=1e-5,(fold,seed,delta)
   np.savez_compressed(dest,prediction=pred.reshape(len(specs),24),specs=np.array([json.dumps(s) for s in specs]),maxdiff=delta,context_row_id=context);print('EC_CPU_PFN_DONE',fold,seed,'maxdiff',delta,'seconds',round(time.time()-start),flush=True);del model;gc.collect()
 print('EC_CPU_PFN_ALL_DONE',flush=True)
if __name__=='__main__':main()
