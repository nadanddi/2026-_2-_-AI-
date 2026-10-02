from support import *
import time,gc
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'))
import env_extra_gpu,torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

def main():
    torch.set_num_threads(4);assert torch.cuda.is_available();start=time.time()
    # Consume each prepared fold immediately; CPU preparation can continue independently.
    prefixes=[f'T_DIAG10_{i}' for i in range(10)]+['T_EXT10_0','T_EXT12_0']
    prefixes += [f'E_{name}_{i}' for name,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for i in range(n)]
    for prefix in prefixes:
        ip=OUT/f'{prefix}_input.npz'
        while not ip.exists():time.sleep(2)
        z=dict(np.load(ip));temp=prefix.startswith('T_');seeds=PSEEDS if temp else [1,2,3,4]
        for seed in seeds:
            cp=OUT/f'{prefix}_pfn_{seed}.npz'
            if cp.exists():continue
            p=z['w']/z['w'].sum() if temp else None
            ix=np.random.default_rng(seed).choice(len(z['x']),min(2000,len(z['x'])),replace=False,p=p)
            model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
            model.fit(z['x'][ix],z['y'][ix]);pred=model.predict(z['q'])
            assert np.isfinite(pred).all()
            np.savez(cp,row_id=z['query_id'],context_row_id=z['train_id'][ix],prediction=pred,input_hash=np.array(sha(ip)))
            del model;gc.collect();torch.cuda.empty_cache()
            print(f'{prefix}/PFN{seed} done elapsed{time.time()-start:.0f}s',flush=True)
    # Repeat a complete PFN fit on the first unchanged context, not just prediction.
    inp=OUT/'T_DIAG10_0_input.npz';z=dict(np.load(inp));cp=dict(np.load(OUT/'T_DIAG10_0_pfn_1.npz'));ix=np.random.default_rng(1).choice(len(z['x']),min(2000,len(z['x'])),replace=False,p=z['w']/z['w'].sum());assert np.array_equal(z['train_id'][ix],cp['context_row_id'])
    replay=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',n_estimators=4,random_state=1,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1);replay.fit(z['x'][ix],z['y'][ix]);p=replay.predict(z['q']);difference=float(np.max(np.abs(p-cp['prediction'])));assert difference<1e-5,difference
    savej(OUT/'gpu_done.json',dict(elapsed=time.time()-start,code_hash=sha(HERE/'gpu.py'),torch=str(torch.__version__),first_context_complete_retrain_maxdiff=difference))
if __name__=='__main__':main()
