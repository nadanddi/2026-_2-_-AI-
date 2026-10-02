from pathlib import Path
import sys, os, json, hashlib, importlib.util
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/temp_season_20261002_v1/reference'))
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/ec_submission10_season_20261002_v1'))
import numpy as np,pandas as pd
import common,harness,temp_mask_v1 as TM,train_flags_v6 as TF,cold_v5
from screen_v6 import temp_members
from resid_reset_features import build_features,FEATURE_COLUMNS
from anal_q1_errors import diag_folds
from season import vectors,mapping
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/statistical_experiments_20261003_v1';OUT.mkdir(parents=True,exist_ok=True)
TSEEDS=[7,101];PSEEDS=list(range(1,9))+list(range(17,25));ESEEDS=[7,101,2024]
K=32;ALPHA=.025/K
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savej(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def safeload():
    tx=pd.read_csv(Path(env.DATA)/'train_X.csv');ty=pd.read_csv(Path(env.DATA)/'train_y.csv',usecols=['row_id','sub_temp']);ty['sub_ec']=np.nan
    sx=pd.read_csv(Path(env.DATA)/'test_X.csv',usecols=['row_id'])
    for c in tx.columns:
        if c!='row_id':sx[c]=np.nan
    for d in (tx,ty,sx):
        d['farm']=d.row_id.str[:3];d['day']=d.row_id.str[4:7].astype(int);d['hour']=d.row_id.str[8:10].astype(int);d['t']=d.day*24+d.hour
    return tx,ty,sx
def loadtemp():
    common.load_raw=safeload
    try:lab,ct,phc=TM.build_world()
    finally:harness._CACHE.clear()
    tx,_,sx=safeload();cf=build_features(tx,sx).set_index('row_id')
    for c in FEATURE_COLUMNS:
        if c not in lab:lab[c]=cf.loc[lab.row_id,c].to_numpy()
    # FULL input never enters fit; only the original EXT definition.
    original=common.load_raw;common.load_raw=TM.ORIG
    # Reuse the already published validation rows instead of reading full labels/test values.
    outer=pd.read_csv(ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv')
    common.load_raw=original
    sets=[]
    for name in ['DIAG10','EXT10','EXT12']:
        if name=='DIAG10':fs=diag_folds(lab)
        else:
            d=outer[(outer.validator==name)&(outer.member=='W30G')&(outer.base_seed==7)&(outer.context=='1-8')]
            fs=[{f:set(d.loc[d.farm==f,'day'].astype(int)) for f in common.TARGET_FARMS}]
        sets.extend((name,k,fd) for k,fd in enumerate(fs))
    return lab,ct,phc,TF.row_weights(lab,.2,w_noisy=.2),sets,outer
def loadcore():
    spec=importlib.util.spec_from_file_location('stat_ec_core',ROOT/'집/코덱스/analysis/codex_independent/rl_ec_v1/run.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);return core
def loadec():
    core=loadcore();raw=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+['out_temp','out_hum','out_rad','out_wspd']+core.RAW);raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
    outer=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv')
    public=outer[(outer.validator=='DIAG10')&(outer.seed==7)][['row_id','sub_ec','block']].copy();assert len(public)==8640
    lab=core.features(raw).merge(public,on='row_id',validate='one_to_one');full=core.identify(raw)
    excluded=set(full[['farm','day']].itertuples(index=False,name=None))-set(lab[['farm','day']].itertuples(index=False,name=None))
    folds=[]
    for (name,k),d in outer[outer.seed==7].groupby(['validator','validation_fold'],sort=False):
        days=set(d[['farm','day']].itertuples(index=False,name=None));banned={(f,int(day)+v) for f,day in days|excluded for v in [-1,0,1]}
        tm=np.array([(f,int(day)) not in banned for f,day in zip(lab.farm,lab.day)]);vm=lab.row_id.isin(d.row_id).to_numpy();assert not (tm&vm).any();folds.append((name,int(k),tm,vm))
    return lab,core,vectors(raw),folds,outer
def inner(tr):
    selected=set()
    for f,d in tr.groupby('farm'):
        days=sorted(d.day.unique());selected|={(f,int(day)) for i,day in enumerate(days) if (i//5)%4==0}
    banned={(f,d+v) for f,d in selected for v in [-1,0,1]}
    tm=np.array([(f,int(d)) not in banned for f,d in zip(tr.farm,tr.day)]);vm=np.array([(f,int(d)) in selected for f,d in zip(tr.farm,tr.day)]);assert not (tm&vm).any();return tm,vm
def seasonal(a,b,wv):
    td=a[['farm','day']].drop_duplicates();qd=b[['farm','day']].drop_duplicates();keys=set(td.itertuples(index=False,name=None));sv={k:wv[k] for k in keys};s,q,_=mapping(td,qd,sv);qs=dict(zip(qd.itertuples(index=False,name=None),q));a=a.copy();b=b.copy();a['season']=[s[(f,int(d))] for f,d in zip(a.farm,a.day)];b['season']=[qs[(f,int(d))] for f,d in zip(b.farm,b.day)];return a,b
def gate(d):return np.where(np.isnan(d.in_temp),1.,np.clip((d.in_temp.to_numpy()-8)/2,0,1))
def weights(d):
    g=gate(d);return np.column_stack([.4+.1*g,.6-.4*g,.3*g])
def conditions(d):
    return np.column_stack([(d.farm=='F47').astype(float),(d.day>=179).astype(float),d.in_temp.to_numpy(),d.in_temp_std.to_numpy(),(d.in_temp_reset3-d.in_temp).to_numpy(),(d.out_temp-d.in_temp).to_numpy(),d.act_heating.to_numpy(),d.out_rad.to_numpy()])
def state_features(observed,query,pred,delay=0):
    # Target aggregates exclusively on observed training days strictly before query.
    daily=observed.groupby(['farm','day'],sort=True).sub_ec.mean();result=[];gaps=[];sources=[]
    for f,day,p in zip(query.farm,query.day,pred):
        choices=[(int(d),float(y)) for (ff,d),y in daily.items() if ff==f and int(d)<int(day)-delay and (int(d)>=179)==(int(day)>=179)]
        if not choices:result.append([0.,0.]);gaps.append(np.nan);sources.append(-1);continue
        d,y=choices[-1];gap=int(day)-d;decay=np.exp(-gap/6);slope=0.
        if len(choices)>1:
            d0,y0=choices[-2];slope=(y-y0)/(d-d0)
        result.append([decay*(y-float(p)),decay*slope]);gaps.append(gap);sources.append(d)
    return np.asarray(result),np.asarray(gaps),np.asarray(sources)
def gapstats(observed,query):
    _,g,s=state_features(observed,query,np.zeros(len(query)));good=g[np.isfinite(g)];days=query[['farm','day']].drop_duplicates();runs=[]
    for f,d in days.groupby('farm'):
        a=sorted(d.day.unique());n=0;last=-100
        for day in a:
            if day!=last+1:
                if n:runs.append(n)
                n=1
            else:n+=1
            last=day
        if n:runs.append(n)
    return dict(n_rows=len(query),n_days=len(days),no_past_rows=int(np.isnan(g).sum()),gap_median=float(np.median(good)) if len(good) else None,gap_p90=float(np.quantile(good,.9)) if len(good) else None,max_missing_day_run=max(runs),train_days=len(observed[['farm','day']].drop_duplicates()))
