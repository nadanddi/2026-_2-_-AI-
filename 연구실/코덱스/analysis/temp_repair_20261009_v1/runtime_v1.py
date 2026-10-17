"""Official-data-only runtime; immutable package imports, no EC numeric labels."""
from pathlib import Path
import sys,os,json,hashlib,importlib.util
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
P=json.loads((ROOT/'연구실/코덱스/analysis/temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
CODE=Path(P['temp_code']); DATA=CODE.parent/'data'
OUT=ROOT/'연구실/코덱스/local/temp_repair_20261009_v1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
os.environ['AGRI_DATA']=str(DATA)
sys.path.insert(0,str(ROOT/'.analysis-tools/python'));sys.path.append(str(ROOT/'.analysis-tools/extra'))
DLL=[]
for d in [ROOT/'.analysis-tools/msvc',ROOT/'.analysis-tools/extra/torch/lib']:
    if d.exists() and hasattr(os,'add_dll_directory'):DLL.append(os.add_dll_directory(str(d)))
sys.path.insert(0,str(CODE))
import env
import numpy as np,pandas as pd
import common,harness

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def meta(d):
    d=d.copy();d['farm']=d.row_id.str[:3];d['day']=d.row_id.str[4:7].astype(int)
    d['hour']=d.row_id.str[8:10].astype(int);d['t']=24*d.day+d.hour
    return d
RAW=(meta(pd.read_csv(DATA/'train_X.csv')),meta(pd.read_csv(DATA/'train_y.csv',usecols=['row_id','sub_temp'])),meta(pd.read_csv(DATA/'test_X.csv')))
RAW[1]['sub_ec']=np.nan

def loader(raw=None):return tuple(x.copy(deep=True) for x in (RAW if raw is None else raw))
common.load_raw=loader
import temp_mask_v1 as TM,train_flags_v6 as TF,make_submission_v13_temp as V
from resid_reset_features import build_features,FEATURE_COLUMNS,PHYSICS_COLUMNS
from anal_q1_errors import diag_folds
spec=importlib.util.spec_from_file_location('repair_season',CODE/'season_ec/season.py')
SEASON=importlib.util.module_from_spec(spec);spec.loader.exec_module(SEASON)
CO2=[c for c in FEATURE_COLUMNS if c=='in_co2' or c.startswith('in_co2_')]
assert len(CO2)==5 and not set(CO2)&set(PHYSICS_COLUMNS)

def masks(raw):
    tx,ty,sx=loader(raw);sx[common.USABLE]=np.nan
    return tx,ty,sx

def median3(df):
    a=meta(df);r=a.copy()
    for _,g in a.groupby(['farm','day']):
        s=g.set_index('hour').in_co2.reindex(range(24)).rolling(3,min_periods=1).median()
        r.loc[g.index,'in_co2']=s.loc[g.hour].to_numpy()
    return r

def cframes(raw=None,smooth=False):
    tx,_,sx=masks(RAW if raw is None else raw)
    if smooth:tx=median3(tx);sx=median3(sx)
    return build_features(tx,sx).set_index('row_id')

def eq(a,b,cols):
    av=a[cols].to_numpy(float);bv=b[cols].to_numpy(float)
    if not np.allclose(av,bv,rtol=0,atol=0,equal_nan=True):
        bad=np.argwhere(~np.isclose(av,bv,rtol=0,atol=0,equal_nan=True));raise AssertionError((len(bad),bad[:5].tolist()))
    return int(av.size)

def weights(tr):
    ids=set(tr.row_id);tx,ty,sx=loader()
    tx=tx[tx.row_id.isin(ids)].copy()
    common.load_raw=lambda:(tx.copy(),ty.copy(),sx.copy())
    try:return TF.row_weights(tr,.2,w_noisy=.2)
    finally:common.load_raw=loader

def season_frames(tr,va):
    td=tr[['farm','day']].drop_duplicates();qd=va[['farm','day']].drop_duplicates()
    keys=set(td.itertuples(index=False,name=None));raw=RAW[0]
    selected=raw[[(f,d) in keys for f,d in zip(raw.farm,raw.day)]]
    vec=SEASON.vectors(selected)
    sm,q,notes=SEASON.mapping(td,qd,vec)
    qm=dict(zip(qd.itertuples(index=False,name=None),q))
    a=tr.copy();b=va.copy()
    a['day']=[sm[(f,int(d))] for f,d in zip(tr.farm,tr.day)]
    b['day']=[qm[(f,int(d))] for f,d in zip(va.farm,va.day)]
    return a,b,notes
