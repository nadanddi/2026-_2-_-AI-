from world import *
import time,gc,os
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'))
import env_extra_gpu,torch,tabpfn
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
SEEDS=list(range(1,9))+list(range(17,25))
def main():
    torch.set_num_threads(4);assert torch.cuda.is_available();start=time.time()
    lab,pfn,ct,phc,wb,wp,wv,sets,z=worlds();X=pfn[FEATURE_COLUMNS].to_numpy(np.float32);y=pfn.sub_temp.to_numpy();ixday=FEATURE_COLUMNS.index('day')
    for name,folds in sets:
        for k,fd in enumerate(folds):
            tm,vm=common.split_mask(pfn,fd);tr,va=season_fold(pfn[tm],pfn[vm],wv,name,k)
            XS=X[tm].copy();VS=X[vm].copy();XS[:,ixday]=tr.season;VS[:,ixday]=va.season
            for seed in SEEDS:
                path=OUT/f'pfn_{name}_{k}_{seed}.npz'
                if path.exists():continue
                idx=np.random.default_rng(seed).choice(int(tm.sum()),min(2000,int(tm.sum())),replace=False,p=wp[tm]/wp[tm].sum());payload={'row_id':va.row_id.to_numpy(dtype=str),'context_row_id':tr.row_id.iloc[idx].to_numpy(dtype=str)}
                for tag,xt,xv in [('base',X[tm],X[vm]),('season',XS,VS)]:
                    model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32)
                    model.fit(xt[idx],y[tm][idx]);payload[tag]=model.predict(xv)
                    del model;gc.collect();torch.cuda.empty_cache()
                np.savez(path,**payload)
                print(f'{name}/{k}/seed{seed} {rmse(payload["base"],y[vm]):.5f}->{rmse(payload["season"],y[vm]):.5f} elapsed{time.time()-start:.0f}s',flush=True)
    (OUT/'pfn_done.json').write_text(json.dumps(dict(elapsed=time.time()-start,seeds=SEEDS,torch=str(torch.__version__),tabpfn=tabpfn.__file__,code_hash=sha(HERE/'tk2_pfn.py')),indent=2),encoding='utf-8')
if __name__=='__main__':main()
