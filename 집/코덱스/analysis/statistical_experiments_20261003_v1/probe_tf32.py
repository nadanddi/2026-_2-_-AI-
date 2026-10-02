from support import *
import gc,time
sys.path.insert(0,str(ROOT/'.analysis-tools/extra_gpu'));sys.path.insert(1,str(ROOT/'.analysis-tools/extra'))
import env_extra_gpu,torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
torch.set_num_threads(4)
def main():
    z=dict(np.load(OUT/'T_DIAG10_0_input.npz'));cp=dict(np.load(OUT/'T_DIAG10_0_pfn_1.npz'));ix=np.random.default_rng(1).choice(len(z['x']),min(2000,len(z['x'])),replace=False,p=z['w']/z['w'].sum());result={}
    for precision in ['highest','high']:
        torch.set_float32_matmul_precision(precision);start=time.time();m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',n_estimators=4,random_state=1,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1);m.fit(z['x'][ix],z['y'][ix]);pred=m.predict(z['q']);result[precision]=dict(seconds=time.time()-start,max_difference=float(np.max(np.abs(pred-cp['prediction']))));del m;gc.collect();torch.cuda.empty_cache();print(precision,result[precision],flush=True)
    result['pass']=result['highest']['max_difference']<1e-5 and result['high']['max_difference']<1e-5 and result['high']['seconds']<result['highest']['seconds']/1.5
    savej(HERE/'tf32_probe.json',result)
if __name__=='__main__':main()
