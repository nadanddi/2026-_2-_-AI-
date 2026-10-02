from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/temp_season_20261002_v1/reference'))
import numpy as np,pandas as pd
import common,harness,temp_mask_v1 as TM,train_flags_v6 as TF,cold_v5
from anal_q1_errors import diag_folds
from screen_v6 import temp_members
from resid_reset_features import build_features,FEATURE_COLUMNS
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/ec_submission10_season_20261002_v1'))
from season import vectors,mapping
OUT=ROOT/'집/코덱스/local/temp_tk_season_20261003_v1';OUT.mkdir(parents=True,exist_ok=True)
def rmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worlds():
    common.load_raw=TM.masked_loader
    try:lab,ct,phc=TM.build_world()
    finally:common.load_raw=TM.ORIG;harness._CACHE.clear()
    tx,ty,sx=TM.masked_loader();cf=build_features(tx,sx).drop(columns=['farm','day','hour','t']).set_index('row_id')
    pfn=lab[['row_id','farm','day','hour','t','sub_temp']].join(cf,on='row_id')
    for c in FEATURE_COLUMNS:
        if c not in lab:lab[c]=cf.loc[lab.row_id,c].to_numpy()
    full,_,fullsx=TM.ORIG();ff=build_features(full,fullsx).set_index('row_id')
    maxdiff=float(np.nanmax(np.abs(pfn[FEATURE_COLUMNS].to_numpy(float)-ff.loc[pfn.row_id,FEATURE_COLUMNS].to_numpy(float))))
    assert maxdiff==0
    z=dict(np.load(Path(env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True));assert np.array_equal(lab.row_id,z['row_id'])
    # FULL only for original EXT day definitions, never model features.
    labF,_,_=TM.build_world();dmin=labF.groupby(['farm','day']).ph_in_temp_3.min()
    sets=[('DIAG10',diag_folds(lab))]
    for th in (10,12):
        ix=dmin[dmin<th].index;sets.append((f'EXT{th}',[{f:set(int(d) for ff,d in ix if ff==f) for f in common.TARGET_FARMS}]))
    el=[]
    for f in common.TARGET_FARMS:
        days=sorted(lab.loc[(lab.farm==f)&(lab.day>=179),'day'].unique())
        for k in range(0,len(days),5):el.append({ff:set(map(int,days[k:k+5])) if ff==f else set() for ff in common.TARGET_FARMS})
    sets.append(('EL1',el))
    wv=vectors(full[full.farm.isin(common.TARGET_FARMS)])
    audit=dict(pfn_columns=FEATURE_COLUMNS,base_columns=ct,hinge_columns=[c for c in ct if 'day_hinge' in c],pfn_full_mask_train_maxdiff=maxdiff,pfn_cache_shapes={s:{v:list(np.load(Path(env.LOCAL)/f'web_tabpfn_{v}_temp_{s}.npy').shape) for v in ['v2','v6']} for s in ['DIAG10','EXT10','EXT12']},validators={s:len(fs) for s,fs in sets},late_days=int(lab.loc[lab.day>=179,['farm','day']].drop_duplicates().shape[0]))
    (OUT/'world_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return lab,pfn,ct,phc,TF.row_weights(lab,.2,w_noisy=.2),TF.row_weights(pfn,.2,w_noisy=.2),wv,sets,z
def season_fold(tr,va,wv,name,k):
    td=tr[['farm','day']].drop_duplicates();qd=va[['farm','day']].drop_duplicates()
    keys=set(td.itertuples(index=False,name=None));v={key:wv[key] for key in keys}
    assert not keys&set(qd.itertuples(index=False,name=None))
    sm,q,notes=mapping(td,qd,v);qm=dict(zip(qd.itertuples(index=False,name=None),q))
    a=tr.copy();b=va.copy()
    a['season']=[sm[(f,int(d))] for f,d in zip(a.farm,a.day)];b['season']=[qm[(f,int(d))] for f,d in zip(b.farm,b.day)]
    notes['train_days']=[list(x) for x in sorted(keys)];notes['query_days']=[list(x) for x in qd.itertuples(index=False,name=None)]
    (OUT/f'{name}_{k}_season.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    return a,b
def base_predict(tr,va,w,ct,phc,seed):
    cold_v5.SEED=seed;M=temp_members(tr,va.reset_index(drop=True),ct,phc,w)
    return .65*M['res']+.25*M['ridge']+.10*M['nys']
def w30(b,c,p,g):return (.4+.1*g)*b+(.6-.4*g)*c+.3*g*p
