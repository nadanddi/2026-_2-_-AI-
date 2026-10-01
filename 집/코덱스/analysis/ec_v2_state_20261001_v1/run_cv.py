from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import ec_model as m
import importlib.util,json,gc,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;OUT=ROOT/'집/코덱스/local/ec_v2_state_20261001_v1';OUT.mkdir(exist_ok=True,parents=True)
BPATH=ROOT/'집/코덱스/analysis/ec_restart_phase3_20261001_v1/run_benchmark.py'
spec=importlib.util.spec_from_file_location('readonly_prior_ec_benchmark',BPATH);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    manifest={'model_sha':m.sha(HERE/'ec_model.py'),'cv_sha':m.sha(__file__),'protocol_sha':m.sha(HERE/'PROTOCOL.md'),'baseline_code_sha':m.sha(BPATH),'baseline_manifest_sha':m.sha(b.OUT/'manifest.json'),'inputs':{n:m.sha(b.DATA/n) for n in ['train_X.csv','train_y.csv']},'lock':m.sha(b.LOCK)}
    if (OUT/'cv_manifest.json').exists():assert json.loads((OUT/'cv_manifest.json').read_text(encoding='utf-8'))==manifest
    else:save(OUT/'cv_manifest.json',manifest)
    raw,full,lab,lock,sigs,fds=b.prepare();rows=[];scores=[];start=time.perf_counter()
    for name,i,vd in fds:
        va=lab[[(f,int(d)) in vd for f,d in zip(lab.farm,lab.day)]].reset_index(drop=True);tr=lab[b.near_mask(lab,vd|lock)].reset_index(drop=True)
        assert set(tr.row_id).isdisjoint(set(va.row_id));assert set(tr[['farm','day']].itertuples(index=False,name=None)).isdisjoint(lock|vd)
        m.state_checks(va,tr);st=m.state_features(tr,tr);sq=m.state_features(va,tr)
        old=b.OUT/f'{name}_{i}.npz';meta=json.loads(old.with_suffix('.json').read_text(encoding='utf-8'));assert m.sha(old)==meta['prediction_sha256']
        with np.load(old,allow_pickle=False) as z:
            assert z['row_id'].tolist()==va.row_id.tolist();p={k:z[k].copy() for k in z.files if k!='row_id'}
        path=OUT/f'{name}_{i}_state.npz'
        if path.exists():
            md=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert md['model_sha']==manifest['model_sha'] and md['baseline_sha']==m.sha(old) and md['npz_sha']==m.sha(path)
            with np.load(path,allow_pickle=False) as z:
                assert z['row_id'].tolist()==va.row_id.tolist();c={s:z[f'candidate_{s}'].copy() for s in m.SEEDS}
        else:
            c={};errs={}
            for s in m.SEEDS:
                e=m.final(m.fit_predict(m.et(s),tr,va,m.FULL),tr,va)
                np.testing.assert_allclose(e,p[f'full_et_{s}'],rtol=0,atol=1e-12)
                es=m.final(m.fit_predict(m.et(s),st,sq,m.FULL+m.STATE),tr,va)
                c[s]=np.clip(p[f'v2_{s}']+.48*(es-e),tr.sub_ec.min(),tr.sub_ec.max())
                print(f'{name}/{i} state seed{s} ready elapsed{time.perf_counter()-start:.0f}s',flush=True)
            np.savez(path,row_id=va.row_id.to_numpy(str),**{f'candidate_{s}':v for s,v in c.items()})
            save(path.with_suffix('.json'),{'model_sha':manifest['model_sha'],'baseline_sha':m.sha(old),'npz_sha':m.sha(path),'state_checks':'PASS','prior_missing_validation':int(sq.prev_public_ec.isna().sum())})
        rec=va[['row_id','farm','day','hour','block','sub_ec']].copy();rec['validator']=name;rec['fold']=i;rec['baseline']=p['v2'];rec['candidate']=np.mean(list(c.values()),axis=0)
        for s in m.SEEDS:
            rec[f'baseline_{s}']=p[f'v2_{s}'];rec[f'candidate_{s}']=c[s]
            ba=b.score(va.sub_ec,p[f'v2_{s}']);ca=b.score(va.sub_ec,c[s]);scores.append({'validator':name,'fold':i,'seed':s,'baseline_rmse':ba,'candidate_rmse':ca,'delta':ca-ba,'n':len(va)})
        rows.append(rec);print(f'DONE {name}/{i} baseline{b.score(va.sub_ec,p["v2"]):.6f} state{b.score(va.sub_ec,rec.candidate):.6f}',flush=True)
    pd.concat(rows,ignore_index=True).to_csv(OUT/'cv_predictions.csv',index=False,float_format='%.17g',encoding='utf-8-sig');pd.DataFrame(scores).to_csv(OUT/'fold_scores.csv',index=False,encoding='utf-8-sig');save(OUT/'cv_complete.json',{'status':'PASS','folds':len(rows),'state_checks':'PASS','old_ET_reproduction':'PASS','final_lock_scored':False,'elapsed_seconds':time.perf_counter()-start})
if __name__=='__main__':
    m.torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
