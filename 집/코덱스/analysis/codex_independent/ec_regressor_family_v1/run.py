"""Controlled regression family comparison on public EC OOF folds."""
import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import importlib.util, json, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

HERE=Path(__file__).resolve().parent
LOCAL=ROOT/'집/코덱스/analysis/local'
H12=HERE.parent/'ec_chain_causal_h12/run_v2.py'
spec=importlib.util.spec_from_file_location('ec_family_features',H12)
h12=importlib.util.module_from_spec(spec);spec.loader.exec_module(h12)
DATA=Path(env.DATA)
SPLITS=LOCAL/'rl_ec_v1/20260927_173801/splits.csv'
LOCK=HERE.parent/'ec_final_lock/locked_days.json'
SEARCH=LOCAL/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=LOCAL/'ec_locked_confirmation/20260928_044934'
OUT=LOCAL/'ec_regressor_family_v1'/datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True,exist_ok=False)
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [HERE/'PROTOCOL.md',HERE/'run.py',H12,DATA/'train_X.csv',DATA/'train_y.csv',SPLITS,LOCK]}
raw=pd.read_csv(DATA/'train_X.csv')
raw=raw[raw.row_id.str[:3].isin(('F13','F47'))].reset_index(drop=True)
lab=h12.features(raw)[['row_id','farm','hour',*h12.FULL]]
lab=lab.merge(pd.read_csv(DATA/'train_y.csv',usecols=['row_id','sub_ec']),on='row_id',validate='one_to_one')
lab=lab.merge(pd.read_csv(SPLITS)[['row_id','fold']],on='row_id',validate='one_to_one')
assert len(lab)==9600 and lab.row_id.is_unique
lock={(r['farm'],int(r['day'])) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
all_val={(f,int(d)) for f,d in lab.loc[lab.fold.isin((8,9)),['farm','day']].itertuples(index=False,name=None)}
dev=lab[h12.split_mask(lab,all_val)].copy()
records=[]
for fold in (0,2,4,6,8,9):
    saved=pd.read_csv((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
    va=lab.set_index('row_id').loc[saved.row_id].reset_index()
    np.testing.assert_allclose(va.sub_ec,saved.sub_ec,rtol=0,atol=1e-12)
    days={(f,int(d)) for f,d in va[['farm','day']].itertuples(index=False,name=None)}
    assert not days & lock
    tr=dev[h12.split_mask(dev,days|lock)].copy() if fold<8 else dev[h12.split_mask(dev,lock)].copy()
    v2=saved['blend' if fold<8 else 'candidate'].to_numpy(float)
    y=va.sub_ec.to_numpy(float)
    for seed in (7,101):
        for name in ('rf','hgb'):
            if name=='rf':
                fill=SimpleImputer(strategy='median',keep_empty_features=True)
                xt=fill.fit_transform(tr[h12.FULL]); xv=fill.transform(va[h12.FULL])
                model=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=seed)
            else:
                xt=tr[h12.FULL].to_numpy(float); xv=va[h12.FULL].to_numpy(float)
                model=HistGradientBoostingRegressor(max_iter=250,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,random_state=seed)
            model.fit(xt,tr.sub_ec.to_numpy(float))
            pred=model.predict(xv)
            combo=.8*v2+.2*pred
            z=va[['row_id','farm','day','hour','sub_ec']].copy()
            z['fold']=fold;z['seed']=seed;z['family']=name;z['v2']=v2;z['model']=pred;z['combo']=combo
            z.to_csv(OUT/f'{name}_fold{fold}_seed{seed}.csv',index=False,float_format='%.17g')
            def score(g,col): return float(np.sqrt(np.mean((g.sub_ec-g[col])**2)))
            for farm,g in [('all',z),*list(z.groupby('farm'))]:
                records.append(dict(family=name,fold=fold,seed=seed,farm=farm,rows=len(g),v2=score(g,'v2'),standalone=score(g,'model'),blend=score(g,'combo'),relative_change=score(g,'combo')/score(g,'v2')-1))
            print(name,fold,seed,records[-3]['relative_change'],flush=True)
pd.DataFrame(records).to_csv(OUT/'scores.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps(dict(hashes=hashes,folds=[0,2,4,6,8,9],seeds=[7,101],lock_days=len(lock),train_rows=len(lab)),indent=2),encoding='utf-8')
print('OUTPUT',OUT)
