from pathlib import Path
import sys, os
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import env_extra
for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']: os.environ[key]='1'
sys.dont_write_bytecode=True
import csv, json, hashlib, importlib.util, time, math, platform
import numpy as np
import pandas as pd
import torch, tabpfn, sklearn, lightgbm
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=ROOT/'집'/'코덱스'/'local'/'ec_restart_phase3_20261001_v1'
OUT.mkdir(parents=True,exist_ok=True)
DATA=Path(env.DATA)
CORE=Path(env.CODEX)/'rl_ec_v1'/'run.py'
spec=importlib.util.spec_from_file_location('readonly_ec_core_phase3',CORE)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
RAW14=['out_temp','out_hum','out_rad','out_wspd']+core.RAW
RAW18=RAW14+['day','hr_sin','hr_cos','midnight']
SEEDS=[7,101,2024]
SPLIT=ROOT/'집'/'코덱스'/'analysis'/'local'/'rl_ec_v1'/'20260927_173801'/'splits.csv'
LOCK=Path(env.CODEX)/'ec_final_lock'/'locked_days.json'
CKPT=Path.home()/'AppData'/'Roaming'/'tabpfn'/'tabpfn-v2-regressor.ckpt'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(s):
    print(s,flush=True)
    with (OUT/'progress.log').open('a',encoding='utf-8') as f:f.write(s+'\n')
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def near_mask(d,days):
    forbidden={(f,v+j) for f,v in days for j in [-1,0,1]}
    return np.array([(f,int(v)) not in forbidden for f,v in zip(d.farm,d.day)])
def final(p,tr,va):return np.clip(core.shrink(p,va),tr.sub_ec.min(),tr.sub_ec.max())
def score(y,p):return float(np.sqrt(np.mean((np.array(y)-np.array(p))**2)))

def prepare():
    raw=pd.read_csv(DATA/'train_X.csv',usecols=['row_id']+RAW14)
    raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
    full=core.identify(raw)
    lock={(z['farm'],int(z['day'])) for z in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    records=[]
    with (DATA/'train_y.csv').open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            farm,day,_=r['row_id'].split('_')
            if farm not in ['F13','F47'] or (farm,int(day)) in lock:continue
            records.append((r['row_id'],float(r['sub_ec'])))
    y=pd.DataFrame(records,columns=['row_id','sub_ec'])
    build=core.features(raw)
    build=build.merge(full[['row_id']+RAW14[:4]],on='row_id',validate='one_to_one')
    lab=build.merge(y,on='row_id',validate='one_to_one').merge(pd.read_csv(SPLIT)[['row_id','fold','block']],on='row_id',validate='one_to_one')
    assert len(lab)==8640 and lab.row_id.is_unique
    lab['late']=lab.day.ge(179)
    # Input-only causal smoothing for original EXT definitions.
    temp=full.sort_values(['farm','day','hour']).copy()
    temp['smooth']=temp.groupby('farm').in_temp.transform(lambda s:s.ewm(halflife=3,ignore_na=True).mean())
    dmin=temp.groupby(['farm','day']).smooth.min()
    signatures={}
    for (f,d),g in full.groupby(['farm','day']):
        g=g.sort_values('hour')
        signatures[(f,int(d))]={kind:hashlib.sha256(pd.util.hash_pandas_object(g[cols],index=False).values.tobytes()).hexdigest()
                               for kind,cols in [('weather',RAW14[:4]),('full14',RAW14)]}
    fds=[('DIAG10',i,{(f,int(d)) for f,d in lab.loc[lab.fold.eq(i),['farm','day']].itertuples(index=False,name=None)}) for i in range(10)]
    layout=list(range(5))+list(range(15,25))+list(range(35,45))+list(range(50,55))
    for name,shifts in [('A',[70,85,100,113,126]),('B',[63,77,92,107,120])]:
        for i,s in enumerate(shifts):fds.append((name,i,{(f,s+o) for f in ['F13','F47'] for o in layout}))
    for th in [10,12]:fds.append((f'EXT{th}',0,{(f,int(d)) for (f,d),v in dmin.items() if v<th and (f,int(d)) not in lock}))
    return raw,full,lab,lock,signatures,fds

def features_check(raw):
    old=core.features(raw).set_index('row_id');meta=core.identify(raw)
    pd.testing.assert_frame_equal(old,core.features(raw.sample(frac=1,random_state=61001)).set_index('row_id'))
    for farm in ['F13','F47']:
        day=int(meta.loc[meta.farm.eq(farm),'day'].median());cut=day*24+6
        past=meta.farm.eq(farm)&(meta.day*24+meta.hour).le(cut)
        for mask in [meta.farm.eq(farm)&~past,meta.farm.ne(farm)]:
            new=raw.copy();new.loc[mask,RAW14]=new.loc[mask,RAW14]*11+777
            pd.testing.assert_frame_equal(old.loc[meta.loc[past,'row_id']],core.features(new).set_index('row_id').loc[meta.loc[past,'row_id']])
        pd.testing.assert_frame_equal(old.loc[meta.loc[past,'row_id']],core.features(raw[past]).set_index('row_id').loc[meta.loc[past,'row_id']])
    new=raw.copy();new.loc[new.row_id.str.endswith('_00'),core.RAW]=np.nan
    assert core.features(new)[[c+'_h0' for c in core.RAW]].isna().all().all()
    return 'PASS'

def audit_fold(name,i,tr,va,signatures,original):
    a={'validator':name,'fold':i,'train_rows':len(tr),'val_rows':len(va),'train_days':len(tr[['farm','day']].drop_duplicates()),
       'val_days':len(va[['farm','day']].drop_duplicates()),'val_late_fraction':float(va.late.mean()),'overlap_ids':len(set(tr.row_id)&set(va.row_id))}
    dist=[]
    for f,g in va.groupby('farm'):
        tt=(tr.loc[tr.farm.eq(f),'day']*24+tr.loc[tr.farm.eq(f),'hour']).to_numpy()
        vt=(g.day*24+g.hour).to_numpy();dist.extend(np.min(np.abs(vt[:,None]-tt[None,:]),axis=1).tolist())
        assert min(dist)>=25
    a['gap_hours']={'min':int(min(dist)),'median':float(np.median(dist)),'max':int(max(dist))}
    td=set(tr[['farm','day']].itertuples(index=False,name=None));vd=set(va[['farm','day']].itertuples(index=False,name=None))
    for kind in ['weather','full14']:
        seen={signatures[k][kind] for k in td}
        a[kind+'_duplicate_val_days']=sum(signatures[k][kind] in seen for k in vd)
    return a

def fit_fold(tr,va,name,i):
    p={};train_scores={};check_key=sha(__file__)+sha(CORE)+sha(CKPT)+sha(SPLIT)+sha(LOCK)
    for g in [tr,va]:check_key+=hashlib.sha256(pd.util.hash_pandas_object(g[['row_id','sub_ec']+core.FULL],index=False).values.tobytes()).hexdigest()
    path=OUT/f'{name}_{i}.npz';manifest=path.with_suffix('.json')
    if path.exists():
        assert json.loads(manifest.read_text(encoding='utf-8'))['key']==check_key
        with np.load(path) as z:
            assert z['row_id'].tolist()==va.row_id.tolist()
            p={k:z[k] for k in z.files if k!='row_id'}
        return p,json.loads(manifest.read_text(encoding='utf-8'))['train_scores']
    y=tr.sub_ec.to_numpy(float)
    p['mean']=np.full(len(va),y.mean())
    means=tr.groupby('farm').sub_ec.mean();p['farm_mean']=va.farm.map(means).to_numpy(float)
    m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10))
    m.fit(tr[RAW18],y);p['ridge']=final(m.predict(va[RAW18]),tr,va);train_scores['ridge']=score(y,final(m.predict(tr[RAW18]),tr,tr))
    r3=[]
    for s in SEEDS:
        m=core.et(s);m.fit(tr[RAW18],y);m.steps[-1][1].n_jobs=1
        p[f'raw_et_{s}']=final(m.predict(va[RAW18]),tr,va);train_scores[f'raw_et_{s}']=score(y,final(m.predict(tr[RAW18]),tr,tr))
        # exact same FULL ET member as the fixed R3 recipe; reuse within-fold fit.
        m=core.et(s);m.fit(tr[core.FULL],y);m.steps[-1][1].n_jobs=1
        e=m.predict(va[core.FULL]);p[f'full_et_{s}']=final(e,tr,va);train_scores[f'full_et_{s}']=score(y,final(m.predict(tr[core.FULL]),tr,tr))
        l=core.predict_model(core.lg(s,'tweedie'),tr,va,core.BASE)
        mlp=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),core.MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-2,learning_rate_init=1e-3,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=s))
        ml=core.predict_model(mlp,tr,va,core.BASE)
        r=.6*e+.3*l+.1*ml;r3.append(r);p[f'r3_{s}']=final(r,tr,va)
        log(f'{name}/{i} R3 seed{s} ready')
    Xtr=tr[core.FULL].to_numpy(np.float32);Xva=va[core.FULL].to_numpy(np.float32)
    members=[]
    for s in [1,2,3,4]:
        idx=np.random.default_rng(s).choice(len(tr),size=min(2000,len(tr)),replace=False)
        m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',n_estimators=4,random_state=s,ignore_pretraining_limits=True,inference_precision=torch.float32)
        m.fit(Xtr[idx],y[idx]);pr=np.asarray(m.predict(Xva),float)
        if name=='DIAG10' and i==0 and s==1:
            again=np.asarray(m.predict(Xva[:8]),float);assert np.array_equal(pr[:8],again),'TabPFN repeat failed'
        members.append(pr);log(f'{name}/{i} TabPFN context{s} ready')
    bag=np.mean(members,axis=0)
    for s,r in zip(SEEDS,r3):p[f'v2_{s}']=final(.8*r+.2*bag,tr,va)
    p['r3']=final(np.mean(r3,axis=0),tr,va);p['v2']=final(.8*np.mean(r3,axis=0)+.2*bag,tr,va)
    p['raw_et']=np.mean([p[f'raw_et_{s}'] for s in SEEDS],axis=0)
    p['full_et']=np.mean([p[f'full_et_{s}'] for s in SEEDS],axis=0)
    assert all(np.isfinite(v).all() for v in p.values())
    np.savez(path,row_id=va.row_id.to_numpy(str),**p)
    save(manifest,{'key':check_key,'train_scores':train_scores,'prediction_sha256':sha(path)})
    return p,train_scores

def structural_ids(full,lab,lock):
    te=core.identify(pd.read_csv(DATA/'test_X.csv',usecols=['row_id']))
    assert set(te.row_id).isdisjoint(set(full.row_id))
    r={}
    for name,a in [('original_train',full),('unlocked_train',lab),('lock_purged_train',lab[near_mask(lab,lock)])]:
        dis=[]
        for f,g in te.groupby('farm'):
            tt=(a.loc[a.farm.eq(f),'day']*24+a.loc[a.farm.eq(f),'hour']).to_numpy()
            dis.extend(np.min(np.abs((g.day*24+g.hour).to_numpy()[:,None]-tt[None,:]),axis=1).tolist())
        r[name]={'min_hours':int(min(dis)),'median_hours':float(np.median(dis)),'max_hours':int(max(dis))}
    r['evaluation_ids_rows']=len(te);r['evaluation_feature_values_read']=False
    return r

def main():
    manifest={'code':sha(__file__),'core':sha(CORE),'protocol':sha(HERE/'PROTOCOL.md'),'lock':sha(LOCK),'splits':sha(SPLIT),'checkpoint':sha(CKPT),
              'inputs':{n:sha(DATA/n) for n in ['train_X.csv','train_y.csv']},'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'torch':torch.__version__,'tabpfn':tabpfn.__version__}
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))==manifest
    else:save(OUT/'manifest.json',manifest)
    raw,full,lab,lock,signatures,fds=prepare()
    checks=features_check(raw);save(HERE/'structural_ids.json',structural_ids(full,lab,lock))
    audits=[];rows=[];scores=[]
    start=time.perf_counter()
    for name,i,vd in fds:
        va=lab[[ (f,int(d)) in vd for f,d in zip(lab.farm,lab.day) ]].reset_index(drop=True)
        if va.empty:continue
        tr=lab[near_mask(lab,vd|lock)].reset_index(drop=True)
        a=audit_fold(name,i,tr,va,signatures,full);audits.append(a)
        log(f'START {name}/{i} train{len(tr)} val{len(va)} elapsed{time.perf_counter()-start:.0f}s')
        p,ts=fit_fold(tr,va,name,i)
        rec=va[['row_id','farm','day','hour','block','sub_ec','late']].copy();rec['validator']=name;rec['validation_fold']=i
        for k,v in p.items():rec[k]=v;scores.append({'validator':name,'fold':i,'model':k,'rmse':score(va.sub_ec,v),'train_rmse':ts.get(k),'n':len(va)})
        rows.append(rec);pd.DataFrame(audits).to_json(HERE/'fold_audit.json',orient='records',indent=2,force_ascii=False)
        pd.DataFrame(scores).to_csv(HERE/'fold_scores.csv',index=False,encoding='utf-8-sig')
        log(f'DONE {name}/{i} V2={score(va.sub_ec,p["v2"]):.6f} rawET={score(va.sub_ec,p["raw_et"]):.6f}')
    combined=pd.concat(rows,ignore_index=True)
    combined.to_csv(OUT/'oof_predictions.csv',index=False,float_format='%.17g',encoding='utf-8-sig')
    save(HERE/'completion.json',{'status':'PASS','feature_checks':checks,'folds':len(audits),'local_output':str(OUT),'elapsed_seconds':time.perf_counter()-start,'manifest':manifest,'final_lock_scored':False,'evaluation_predictions_created':False})
    log('BENCHMARK COMPLETE')

if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
