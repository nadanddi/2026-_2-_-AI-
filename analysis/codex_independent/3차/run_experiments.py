# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'.analysis-tools/python'));sys.path.insert(0,str(ROOT/'research'))
import env
import numpy as np
import pandas as pd
import common,ec_v6
from harness import load,folds
import features_v4 as F4
from ec_features import build_features,FEATURE_COLUMNS
from ec_models import fit_predict_all,NAMES,PARAMS

def diag_folds(d):
    fs=[{f:set() for f in ['F13','F47']} for _ in range(10)]
    for f in ['F13','F47']:
        ds=sorted(d.loc[d.farm==f,'day'].unique())
        for i in range(0,len(ds),5):fs[(i//5)%10][f].update(map(int,ds[i:i+5]))
    return fs

def main():
    files=[ROOT/'research/ec_v6.py',ROOT/'research/harness.py',ROOT/'research/features_v4.py',ROOT/'research/fp_features.py',ROOT/'research/make_submission_v3.py',ROOT/'research/Codex_요청_3차_2026-09-26.md',ROOT/'.gitignore',ROOT/'analysis/codex_independent/2차/보고서_과제1.md']+[Path(common.DATA)/f for f in ['train_X.csv','train_y.csv','test_X.csv']]
    hs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (H/'original_hashes.json').write_text(json.dumps(hs,ensure_ascii=False,indent=2),encoding='utf-8')
    (H/'fixed_experiments.json').write_text(json.dumps({'candidates':NAMES,'mix':.2,'params':PARAMS,'validation':['A','DIAG10'],'selection':'A fold mean RMSE; no blend retuning','forbidden':'SP/SP2, outdoor/physics additions, noisy-day downweight, high-value weight, label history'},ensure_ascii=False,indent=2),encoding='utf-8')
    tx,ty,sx=common.load_raw();z=build_features(tx,sx)
    _,_,lab=load();fp=F4.fp_features();fpc=F4.names(fp)
    lab=lab.merge(fp,on='row_id',how='left')
    f14=[c for c in list(common.USABLE)+['day','hr_sin','hr_cos','midnight'] if c not in common.OUT_COLS]
    # 원래 baseline 피처는 그대로 사용, 독립 피처는 별도 테이블에 정렬.
    d=z.set_index('row_id').loc[lab.row_id].reset_index();d['sub_ec']=lab.sub_ec.to_numpy()
    d['closed_day']=d.groupby(['farm','day']).act_circfan.transform('mean').lt(10)&d.groupby(['farm','day']).act_vent.transform(lambda s:(s==0).mean()).gt(.85)
    saved=np.load(ROOT/'research/local/oof_ec_diag.npz',allow_pickle=True)
    baseline_diag=pd.Series(saved['oof'],index=saved['row_id']).loc[d.row_id].to_numpy()
    assert np.isfinite(baseline_diag).all()
    d.to_json(H/'validation_rows.json',orient='records',force_ascii=False,double_precision=15)
    paths=[]
    for kind,fs in [('A',folds('A')),('DIAG10',diag_folds(d))]:
        for i,fd in enumerate(fs):
            tr,va=common.split_mask(d,fd)
            assert not set(map(tuple,d.loc[tr,['farm','day']].values))&set(map(tuple,d.loc[va,['farm','day']].values))
            if kind=='A':
                print('A baseline training',i+1,flush=True)
                b=ec_v6.pipeline(f14+fpc,f14)(lab.loc[tr].reset_index(drop=True),lab.loc[va].reset_index(drop=True))
            else:b=baseline_diag[va]
            pp=fit_predict_all(d.loc[tr].reset_index(drop=True),d.loc[va].reset_index(drop=True))
            assert np.isfinite(b).all() and all(np.isfinite(p).all() for p in pp.values())
            name=f'{kind}_fold{i+1:02d}.npz'
            np.savez_compressed(H/name,row_id=d.loc[va,'row_id'].to_numpy(),baseline=b,**pp)
            paths.append({'kind':kind,'fold':i+1,'file':name,'train_rows':int(tr.sum()),'val_rows':int(va.sum())})
            score=lambda p:float(np.sqrt(np.mean((p-d.loc[va,'sub_ec'].to_numpy())**2)))
            print(kind,i+1,'baseline',score(b),'mix',{k:score(.8*b+.2*p) for k,p in pp.items()},flush=True)
    (H/'fold_files.json').write_text(json.dumps({'folds':paths,'baseline_features':{'rest':f14,'et':f14+fpc},'independent_features':FEATURE_COLUMNS},ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hs.items())
    print('DONE',flush=True)
if __name__=='__main__':main()
