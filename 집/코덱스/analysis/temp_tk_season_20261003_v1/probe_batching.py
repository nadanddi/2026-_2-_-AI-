from world import *
import time,gc
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'))
import env_extra_gpu,torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
torch.set_num_threads(4)
lab,pfn,ct,phc,wb,wp,wv,sets,z=worlds();tm,vm=common.split_mask(pfn,sets[0][1][0]);tr,va=season_fold(pfn[tm],pfn[vm],wv,'PERF',0);idx=np.random.default_rng(1).choice(int(tm.sum()),2000,replace=False,p=wp[tm]/wp[tm].sum());old=dict(np.load(OUT/'pfn_DIAG10_0_1.npz'));stats=[]
for tag in ['base','season']:
    a,b=(tr,va) if tag=='base' else (tr.assign(day=tr.season),va.assign(day=va.season));start=time.time()
    model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',n_estimators=4,random_state=1,ignore_pretraining_limits=True,inference_precision=torch.float32,memory_saving_mode=False)
    model.fit(a[FEATURE_COLUMNS].to_numpy(np.float32)[idx],tr.sub_temp.to_numpy()[idx]);pr=model.predict(b[FEATURE_COLUMNS].to_numpy(np.float32))
    stats.append(dict(variant=tag,elapsed=time.time()-start,maxdiff=float(np.max(np.abs(pr-old[tag])))));del model;gc.collect();torch.cuda.empty_cache()
(HERE/'batching_probe.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print(json.dumps(stats))
