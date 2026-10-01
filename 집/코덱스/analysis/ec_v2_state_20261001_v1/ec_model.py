"""EC-only v2 plus prior-public-EC ExtraTrees delta, fixed recipe."""
from pathlib import Path
import sys,os
HERE=Path(__file__).resolve().parent
ROOT=next((p for p in HERE.parents if (p/'AGENTS.md').exists()),None)
if ROOT:sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
try:import env_extra
except ModuleNotFoundError:pass
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
sys.dont_write_bytecode=True
import csv,json,hashlib,argparse,gc,platform
import numpy as np
import pandas as pd
import sklearn,lightgbm,torch,tabpfn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits
RAW=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
ACTS=RAW[3:];BASE=RAW+['day','hr_sin','hr_cos','midnight']
FULL=BASE+[v+'_h0' for v in ACTS+RAW[:3]]+[n for v in ACTS for n in (v+'_tdm',v+'_tdz')]
STATE=['prev_public_ec','prev_public_gap_hours'];SEEDS=[7,101,2024]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identify(x):
    a=x.copy();a['farm']=a.row_id.str[:3];a['day']=a.row_id.str[4:7].astype(int);a['hour']=a.row_id.str[8:10].astype(int);return a
def features(raw):
    a=identify(raw).sort_values(['farm','day','hour']).reset_index(drop=True)
    assert a.row_id.is_unique
    a['hr_sin']=np.sin(2*np.pi*a.hour/24);a['hr_cos']=np.cos(2*np.pi*a.hour/24);a['midnight']=a.hour.eq(0).astype(float)
    g=a.groupby(['farm','day'],sort=False);h0=a[a.hour.eq(0)].set_index(['farm','day']);key=pd.MultiIndex.from_arrays([a.farm,a.day])
    for v in ACTS+RAW[:3]:a[v+'_h0']=h0[v].reindex(key).values
    for v in ACTS:
        a[v+'_tdm']=g[v].transform(lambda s:s.expanding().mean())
        a[v+'_tdz']=g[v].transform(lambda s:s.eq(0).astype(float).where(s.notna()).expanding().mean())
    return a[['row_id','farm','hour']+FULL]
def near_mask(frame,days):
    forbidden={(f,int(d)+j) for f,d in days for j in [-1,0,1]}
    return np.array([(f,int(d)) not in forbidden for f,d in zip(frame.farm,frame.day)])
def state_features(query,available):
    a=query.copy();daily=available.groupby(['farm','day']).sub_ec.agg(['mean','size']).reset_index();assert daily['size'].eq(24).all()
    ec=np.full(len(a),np.nan);gap=ec.copy();src=ec.copy()
    for farm,idx in a.groupby('farm').indices.items():
        s=daily[daily.farm.eq(farm)].sort_values('day');days=s.day.to_numpy(int)
        q=a.iloc[idx];pos=np.searchsorted(days,q.day.to_numpy(int),side='left')-1;ok=pos>=0
        if not ok.any():continue
        jj=np.asarray(idx)[ok];chosen=days[pos[ok]]
        src[jj]=chosen;ec[jj]=s['mean'].to_numpy(float)[pos[ok]]
        gap[jj]=a.iloc[jj].day.to_numpy(int)*24+a.iloc[jj].hour.to_numpy(int)-(chosen*24+23)
        assert (chosen<a.iloc[jj].day.to_numpy(int)).all() and (gap[jj]>0).all()
    a[STATE[0]]=ec;a[STATE[1]]=gap;a['state_source_day']=src
    allowed=set(available[['farm','day']].itertuples(index=False,name=None))
    assert all((f,int(d)) in allowed for f,d in zip(a.farm,a.state_source_day) if np.isfinite(d))
    return a
def state_checks(query,available):
    ref=state_features(query,available).set_index('row_id')[STATE+['state_source_day']]
    b=state_features(query.sample(frac=1,random_state=883),available.sample(frac=1,random_state=884)).set_index('row_id')[ref.columns]
    pd.testing.assert_frame_equal(ref.sort_index(),b.sort_index())
    for farm,g in query.groupby('farm'):
        q=g[g.day.eq(g.day.median() if g.day.median() in set(g.day) else g.day.iloc[0])].copy()
        if q.empty:q=g.iloc[:1].copy()
        day=int(q.day.iloc[0]);z=available.copy();z.loc[z.farm.ne(farm)|z.day.ge(day),'sub_ec']=999999
        pd.testing.assert_frame_equal(state_features(q,available)[STATE],state_features(q,z)[STATE])
    return 'PASS'
def shrink(p,frame):
    d=frame[['farm','day','hour']].reset_index(drop=True).copy();d['p']=p;d=d.sort_values(['farm','day','hour'])
    avg=d.groupby(['farm','day']).p.transform(lambda s:s.expanding().mean());o=np.empty(len(p));o[d.index.values]=(.5*d.p+.5*avg).values;return o
def final(p,tr,query):return np.clip(shrink(p,query),tr.sub_ec.min(),tr.sub_ec.max())
def et(seed):return make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=600,max_features=1.0,min_samples_leaf=1,n_jobs=4,random_state=seed))
def lg(seed):return lightgbm.LGBMRegressor(n_estimators=800,learning_rate=.03,num_leaves=31,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1,deterministic=True,force_col_wise=True,n_jobs=4,verbose=-1,random_state=seed,objective='tweedie',tweedie_variance_power=1.5)
def mlp(seed):return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-2,learning_rate_init=1e-3,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed))
def fit_predict(model,tr,query,cols):
    model.fit(tr[cols],tr.sub_ec.to_numpy(float))
    if hasattr(model,'steps') and isinstance(model.steps[-1][1],ExtraTreesRegressor):model.steps[-1][1].n_jobs=1
    p=model.predict(query[cols]);del model;gc.collect();return np.asarray(p,float)
def predict_all(tr,query,checkpoint):
    state_checks(query,tr);st=state_features(tr,tr);sq=state_features(query,tr)
    old=[];new=[];r3=[]
    for s in SEEDS:
        e=fit_predict(et(s),tr,query,FULL);old.append(final(e,tr,query))
        es=fit_predict(et(s),st,sq,FULL+STATE);new.append(final(es,tr,query))
        l=fit_predict(lg(s),tr,query,BASE);m=fit_predict(mlp(s),tr,query,BASE)
        r3.append(.6*e+.3*l+.1*m);print(f'FINAL R3/state seed{s} ready',flush=True)
    X=tr[FULL].to_numpy(np.float32);Q=query[FULL].to_numpy(np.float32);y=tr.sub_ec.to_numpy(float);bags=[]
    for s in [1,2,3,4]:
        ix=np.random.default_rng(s).choice(len(tr),size=min(2000,len(tr)),replace=False)
        model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',model_path=str(checkpoint),n_estimators=4,random_state=s,ignore_pretraining_limits=True,inference_precision=torch.float32)
        model.fit(X[ix],y[ix]);p=np.asarray(model.predict(Q),float);bags.append(p)
        if s==1:assert np.array_equal(p[:8],np.asarray(model.predict(Q[:8]),float))
        del model;gc.collect();print(f'FINAL TabPFN context{s} ready',flush=True)
    bag=np.mean(bags,axis=0);base=[final(.8*r+.2*bag,tr,query) for r in r3]
    cand=[np.clip(v+.48*(n-o),tr.sub_ec.min(),tr.sub_ec.max()) for v,n,o in zip(base,new,old)]
    return np.mean(cand,axis=0),sq,{'baseline':final(.8*np.mean(r3,axis=0)+.2*bag,tr,query),'seed_predictions':cand}
def load(data,lock):
    locks={(r['farm'],int(r['day'])) for r in json.loads(Path(lock).read_text(encoding='utf-8'))['selected']}
    x=pd.read_csv(data/'train_X.csv',usecols=['row_id']+RAW);x=x[x.row_id.str[:3].isin(['F13','F47'])]
    tx=pd.read_csv(data/'test_X.csv',usecols=['row_id']+RAW);ids=pd.read_csv(data/'sample_submission.csv',usecols=['row_id'])
    assert tx.row_id.is_unique and ids.row_id.is_unique and set(tx.row_id)==set(ids.row_id) and set(tx.row_id).isdisjoint(set(x.row_id))
    records=[]
    with (data/'train_y.csv').open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            farm,day,_=r['row_id'].split('_')
            if farm not in ['F13','F47'] or (farm,int(day)) in locks:continue
            records.append((r['row_id'],float(r['sub_ec'])))
    y=pd.DataFrame(records,columns=['row_id','sub_ec']);tr=features(x).merge(y,on='row_id',validate='one_to_one')
    tr=tr[near_mask(tr,locks)].reset_index(drop=True)
    # Training features use train inputs only; evaluation features use same-farm causal train+test inputs.
    q=features(pd.concat([x,tx],ignore_index=True)).set_index('row_id').loc[ids.row_id].reset_index()
    assert not any((f,int(d)) in locks for f,d in zip(tr.farm,tr.day))
    return tr,q,ids,locks
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(env.DATA));p.add_argument('--lock',type=Path,default=(ROOT/'집/코덱스/analysis/codex_independent/ec_final_lock/locked_days.json') if ROOT else HERE/'locked_days.json');p.add_argument('--checkpoint',type=Path,default=HERE/'tabpfn-v2-regressor.ckpt' if (HERE/'tabpfn-v2-regressor.ckpt').exists() else Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),'Use a new output directory';a.output.mkdir(parents=True)
    tr,q,ids,locks=load(a.data,a.lock);torch.set_num_threads(4)
    with threadpool_limits(limits=4):pred,state,details=predict_all(tr,q,a.checkpoint)
    assert len(pred)==len(ids)==1440 and np.isfinite(pred).all() and (pred>=0).all()
    answer=pd.DataFrame({'row_id':ids.row_id,'sub_ec':pred});answer.to_csv(a.output/'ec_v2_state_v1.csv',index=False,float_format='%.17g',encoding='utf-8-sig')
    state[['row_id','farm','day','hour']+STATE+['state_source_day']].to_csv(a.output/'past_state_audit.csv',index=False,float_format='%.17g',encoding='utf-8-sig')
    np.savez(a.output/'prediction_details.npz',row_id=ids.row_id.to_numpy(str),baseline=details['baseline'],candidate=pred,**{f'seed_{s}':v for s,v in zip(SEEDS,details['seed_predictions'])})
    r={'status':'PASS','rows':len(answer),'columns':list(answer.columns),'training_rows':len(tr),'training_days':len(tr[['farm','day']].drop_duplicates()),'final_lock_scored':False,'temperature_model_created':False,'candidate_adopted':False,'code_sha256':sha(__file__),'inputs_sha256':{n:sha(a.data/n) for n in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']},'lock_sha256':sha(a.lock),'checkpoint_sha256':sha(a.checkpoint),'csv_sha256':sha(a.output/'ec_v2_state_v1.csv'),'prior_state_missing_rows':int(state.prev_public_ec.isna().sum()),'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'torch':torch.__version__,'tabpfn':tabpfn.__version__}}
    (a.output/'manifest.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
