from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np
OUT=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1';OLD=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
def main():
 OUT.mkdir(parents=True,exist_ok=True);lab,core,wv,folds,outer=S.loadec();checks=0;maximum=0.;audit=[]
 for v,k,tm,vm in folds:
  tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True);payload=dict(row_id=va.row_id.to_numpy(str),lo=tr.sub_ec.min(),hi=tr.sub_ec.max());pp=[]
  for c in [1,2,3,4]:
   z=dict(np.load(OLD/f'{v}_{k}_pfn_{c}.npz'));assert np.array_equal(z['row_id'],va.row_id);assert set(z['context_row_id'])<=set(tr.row_id);assert np.array_equal(z['sub_ec'],va.sub_ec);pp.append(z['raw_pfn']);payload[f'context_{c}']=z['context_row_id'];checks+=3
  bag=np.mean(pp,axis=0);payload['old_pfn_raw']=bag
  for seed in [7,101,2024]:
   z=dict(np.load(OLD/f'{v}_{k}_r3_{seed}.npz'));assert np.array_equal(z['row_id'],va.row_id);ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
   prediction=np.clip(core.shrink(.8*z['raw_r3']+.2*bag,va),tr.sub_ec.min(),tr.sub_ec.max());diff=float(np.max(np.abs(ref-prediction)));assert diff<1e-12;checks+=2;maximum=max(maximum,diff);payload[f'r3_{seed}']=z['raw_r3'];payload[f'baseline_{seed}']=ref
  np.savez(OUT/f'{v}_{k}_baseline.npz',**payload);audit.append(dict(validator=v,fold=k,train_days=len(tr)//24,validation_rows=len(va)))
 result=dict(status='PASS',checks=checks,folds=len(folds),baseline_maxdiff=maximum,audit=audit,scope='Existing public V2 raw-cache and context replay; no new weights downloaded or model fit')
 (H/'baseline_preparation_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
