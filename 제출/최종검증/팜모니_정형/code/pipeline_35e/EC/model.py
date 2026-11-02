"""submission_14 EC = submission_13 (season v2 + DP1 operation-sequence features in the R3
members; TabPFN unchanged) + SG2 post-processing (sg2post.py: guarded control-signature
reference correction on pass-2 rows; catalog 6.310 PASS, 6.311 same direction on the
submission_13 configuration).  Built 2026-10-05 by 집 클로드 per user request ("sg2로만").
sub_temp blank here (teammate temperature merged afterwards)."""
from pathlib import Path
import sys,os
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import env
import env_extra
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
sys.dont_write_bytecode=True
import csv,json,hashlib,argparse,gc,platform,time
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
from season import identify,vectors,mapping,W
import sg2post
RAW=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
ACTS=RAW[3:];DAY_BASE=RAW+['day','hr_sin','hr_cos','midnight']
DAY_FULL=DAY_BASE+[v+'_h0' for v in ACTS+RAW[:3]]+[n for v in ACTS for n in (v+'_tdm',v+'_tdz')]
BASE=[c for c in DAY_BASE if c!='day']+['season']
FULL=[c for c in DAY_FULL if c!='day']+['season']
OPS=['seal_run','vent_open_hours','first_open_hour','thermal_switches','shade_switches','since_curtain_change','heat_run','co2_hours','vent_max']
FULL_R3=FULL+OPS;BASE_R3=BASE+OPS
SEEDS=[7,101,2024]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def run_len(b):
    o=np.zeros(len(b));c=0
    for i,v in enumerate(b):
        c=c+1 if v else 0;o[i]=c
    return o
def ops_day(g):
    v=g.act_vent.fillna(0).to_numpy();th=g.act_thermal.fillna(0).to_numpy();sh=g.act_shade.fillna(0).to_numpy()
    he=g.act_heating.fillna(0).to_numpy();co=g.act_co2.fillna(0).to_numpy();h=g.hour.to_numpy()
    f=pd.DataFrame(index=g.index)
    f['seal_run']=run_len(v==0);f['vent_open_hours']=np.cumsum(v>0)
    f['first_open_hour']=np.where(np.cumsum(v>0)>0,np.minimum.accumulate(np.where(v>0,h,99)),24)
    ts=np.r_[0,np.abs(np.diff((th>0).astype(int)))];ss=np.r_[0,np.abs(np.diff((sh>0).astype(int)))]
    f['thermal_switches']=np.cumsum(ts);f['shade_switches']=np.cumsum(ss)
    last=-1;sc=[]
    for k in range(len(h)):
        if ts[k] or ss[k]:last=k
        sc.append(k-last if last>=0 else k+1)
    f['since_curtain_change']=sc;f['heat_run']=run_len(he>0);f['co2_hours']=np.cumsum(co>0);f['vent_max']=np.maximum.accumulate(v)
    return f
def features(raw):
    a=identify(raw).sort_values(['farm','day','hour']).reset_index(drop=True)
    assert a.row_id.is_unique
    a['hr_sin']=np.sin(2*np.pi*a.hour/24);a['hr_cos']=np.cos(2*np.pi*a.hour/24);a['midnight']=a.hour.eq(0).astype(float)
    g=a.groupby(['farm','day'],sort=False);h0=a[a.hour.eq(0)].set_index(['farm','day']);key=pd.MultiIndex.from_arrays([a.farm,a.day])
    for v in ACTS+RAW[:3]:a[v+'_h0']=h0[v].reindex(key).values
    for v in ACTS:
        a[v+'_tdm']=g[v].transform(lambda s:s.expanding().mean())
        a[v+'_tdz']=g[v].transform(lambda s:s.eq(0).astype(float).where(s.notna()).expanding().mean())
    a=a.join(pd.concat([ops_day(gg) for _,gg in a.groupby(['farm','day'],sort=False)]))
    return a[['row_id','farm','hour']+DAY_FULL+OPS]
def shrink(p,frame):
    d=frame[['farm','day','hour']].reset_index(drop=True).copy();d['p']=p;d=d.sort_values(['farm','day','hour'])
    avg=d.groupby(['farm','day']).p.transform(lambda s:s.expanding().mean());o=np.empty(len(p));o[d.index.values]=(.5*d.p+.5*avg).values;return o
def query_features(x,tx,ids,table):
    q=features(pd.concat([x[['row_id']+RAW],tx[['row_id']+RAW]],ignore_index=True)).set_index('row_id').loc[ids].reset_index()
    q['season']=[table[(f,int(d))] for f,d in zip(q.farm,q.day)]
    return q
def load(data):
    x=pd.read_csv(data/'train_X.csv',usecols=['row_id']+W+RAW)
    assert x.row_id.str[:3].isin(['F13','F47']).all()
    tx=pd.read_csv(data/'test_X.csv',usecols=['row_id']+W+RAW)
    ids=pd.read_csv(data/'sample_submission.csv',usecols=['row_id'])
    assert tx.row_id.is_unique and ids.row_id.is_unique and set(tx.row_id)==set(ids.row_id) and set(tx.row_id).isdisjoint(set(x.row_id))
    y=pd.read_csv(data/'train_y.csv',usecols=['row_id','sub_ec'])
    assert list(y.columns)==['row_id','sub_ec'] and np.isfinite(y.sub_ec).all()
    tr=features(x).merge(y,on='row_id',validate='one_to_one')
    qmeta=identify(tx).sort_values(['farm','day','hour'])
    td=tr[['farm','day']].drop_duplicates();qd=qmeta[['farm','day']].drop_duplicates().reset_index(drop=True)
    ts,qs,notes=mapping(td,qd,vectors(x));table=dict(zip(qd.itertuples(index=False,name=None),qs))
    tr['season']=[ts[(f,int(d))] for f,d in zip(tr.farm,tr.day)]
    # Canonical query order makes public input reordering irrelevant to inference batching.
    q=query_features(x,tx,sorted(ids.row_id),table)
    return x,tx,tr,q,ids,ts,table,notes
def causal_checks(x,tx,q,table):
    baseline=q.set_index('row_id');meta=identify(tx);checks=[]
    shuffled=query_features(x,tx.sample(frac=1,random_state=812),sorted(tx.row_id),table).set_index('row_id')
    pd.testing.assert_frame_equal(baseline,shuffled)
    for farm in ['F13','F47']:
        times=meta.day*24+meta.hour
        for frac in [.2,.5,.8]:
            cut=int(times[meta.farm.eq(farm)].quantile(frac))
            allowed=meta.farm.eq(farm)&times.le(cut)
            changed=tx.copy();changed.loc[~allowed,RAW]=changed.loc[~allowed,RAW]*13+97
            altered=query_features(x,changed,sorted(tx.row_id),table).set_index('row_id')
            keys=meta.loc[allowed,'row_id']
            pd.testing.assert_frame_equal(baseline.loc[keys],altered.loc[keys])
            checks.append({'farm':farm,'cut':cut,'target_rows':len(keys),'future_and_other_farm_features_unchanged':True})
    # Training weather and table are unchanged; query weather is not even loaded.
    return {'input_order_features':'PASS','causal_feature_checks':checks,'evaluation_weather_used_by_model_features':False}
def sg2_checks(x,tx,q,p13,final,ec,ref,lo,hi):
    """Future-input / other-farm / later-prediction perturbation must not change SG2 output of earlier rows."""
    meta=identify(q[['row_id']]).reset_index(drop=True);times=meta.day*24+meta.hour;out=[]
    base=pd.Series(final,index=q.row_id.values)
    tmeta=identify(tx);ttimes=tmeta.day*24+tmeta.hour
    for farm in ['F13','F47']:
        for frac in [.2,.5,.8]:
            cut=int(times[meta.farm.eq(farm)].quantile(frac))
            allowed_q=(meta.farm.eq(farm)&times.le(cut)).values
            allowed_t=(tmeta.farm.eq(farm)&ttimes.le(cut)).values
            changed=tx.copy();cols=[c for c in changed.columns if c!='row_id']
            changed.loc[~allowed_t,cols]=changed.loc[~allowed_t,cols]*13+97
            p=p13.copy();p[~allowed_q]=p[~allowed_q]+1.0
            S2=sg2post.prepare(pd.concat([x,changed],ignore_index=True))
            alt=np.clip(sg2post.correct(q[['row_id']],p,S2,ec,ref,sg2post.ref_calendar(S2,ref)),lo,hi)
            keys=q.row_id.values[allowed_q]
            assert np.array_equal(base.loc[keys].values,pd.Series(alt,index=q.row_id.values).loc[keys].values)
            out.append({'farm':farm,'cut':cut,'target_rows':int(allowed_q.sum()),'sg2_unchanged':True})
            print('SG2 causal check %s cut %d PASS'%(farm,cut),flush=True)
    return {'sg2_future_other_farm_later_prediction_checks':out}
def et(seed):return make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=600,max_features=1.,min_samples_leaf=1,n_jobs=2,random_state=seed))
def lg(seed):return lightgbm.LGBMRegressor(n_estimators=800,learning_rate=.03,num_leaves=31,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1,deterministic=True,force_col_wise=True,n_jobs=2,verbose=-1,random_state=seed,objective='tweedie',tweedie_variance_power=1.5)
def mlp(seed):return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed))
def predict_all(tr,q,checkpoint,out):
    r3=[];contexts=[];pfn=[];raws={};invariance=[]
    for seed in SEEDS:
        predictions=[]
        for name,model,cols in [('et',et(seed),FULL_R3),('lgb',lg(seed),BASE_R3),('mlp',mlp(seed),BASE_R3)]:
            with threadpool_limits(limits=2):
                model.fit(tr[cols],tr.sub_ec.to_numpy(float))
                if name=='et':model.steps[-1][1].n_jobs=1
                pred=np.asarray(model.predict(q[cols]),float)
                assert np.array_equal(pred[:8],np.asarray(model.predict(q.iloc[:8][cols]),float))
            predictions.append(pred);raws[f'{name}_{seed}']=pred;del model;gc.collect()
        r3.append(.6*predictions[0]+.3*predictions[1]+.1*predictions[2])
        print(f'R3 seed{seed} completed',flush=True)
    X=tr[FULL].to_numpy(np.float32);Q=q[FULL].to_numpy(np.float32);y=tr.sub_ec.to_numpy(float)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    for seed in [1,2,3,4]:
        started=time.monotonic();ix=np.random.default_rng(seed).choice(len(tr),size=2000,replace=False)
        model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',model_path=str(checkpoint),n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
        with threadpool_limits(limits=4):
            model.fit(X[ix],y[ix]);p=np.asarray(model.predict(Q),float)
            assert np.array_equal(p[:8],np.asarray(model.predict(Q[:8]),float))
        invariance.append({'context':seed,'target8_full_batch_vs_target_only_bits_equal':True})
        pfn.append(p);contexts.append(tr.row_id.iloc[ix].to_numpy(str))
        np.savez(out/f'context{seed}.npz',row_id=q.row_id.to_numpy(str),raw_pfn=p,context_row_id=contexts[-1])
        del model;gc.collect();print(f'TabPFN context{seed} completed {time.monotonic()-started:.1f}s',flush=True)
    raw_r3=np.mean(r3,axis=0);raw_pfn=np.mean(pfn,axis=0)
    pred=np.clip(shrink(.8*raw_r3+.2*raw_pfn,q),tr.sub_ec.min(),tr.sub_ec.max())
    np.savez(out/'prediction_details.npz',row_id=q.row_id.to_numpy(str),prediction=pred,raw_r3_by_seed=np.stack(r3),raw_pfn_by_context=np.stack(pfn),**raws)
    return pred,invariance
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),'Use a new output directory';a.output.mkdir(parents=True)
    assert np.__version__=='2.5.3'
    assert sha(a.checkpoint)=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    x,tx,tr,q,ids,ts,table,notes=load(a.data)
    assert len(tr)==9600 and len(tr[['farm','day']].drop_duplicates())==400 and len(q)==1440 and len(table)==60
    check=causal_checks(x,tx,q,table)
    train_table=pd.DataFrame([(f,d,v) for (f,d),v in sorted(ts.items())],columns=['farm','day','season'])
    eval_table=pd.DataFrame([(f,d,v) for (f,d),v in sorted(table.items())],columns=['farm','day','season'])
    for name,frame in [('training_day_to_season.csv',train_table),('evaluation_day_to_season.csv',eval_table)]:frame.to_csv(a.output/name,index=False,float_format='%.17g')
    save(a.output/'season_transform.json',notes)
    print('Training 9600 rows / 400 days; evaluation 1440 rows / 60 days; temperature blank',flush=True)
    pred,invariance=predict_all(tr,q,a.checkpoint,a.output)
    pred13=pred.copy()
    y=pd.read_csv(a.data/'train_y.csv',usecols=['row_id','sub_ec']);yi=identify(y)
    ec=yi.groupby(['farm','day']).sub_ec.mean();ref=set(ec.index)
    S=sg2post.prepare(pd.concat([x,tx],ignore_index=True));cal=sg2post.ref_calendar(S,ref)
    lo,hi=float(y.sub_ec.min()),float(y.sub_ec.max())
    pred=np.clip(sg2post.correct(q[['row_id']],pred13,S,ec,ref,cal),lo,hi)
    print('SG2 applied: rows changed %d of %d, mean change %+.4f'%(int((pred!=pred13).sum()),len(pred),float(np.mean(pred-pred13))),flush=True)
    sg2check=sg2_checks(x,tx,q,pred13,pred,ec,ref,lo,hi)
    np.savez(a.output/'sg2_details.npz',row_id=q.row_id.to_numpy(str),pred_submission13_config=pred13,pred_final=pred)
    answer=pd.Series(pred,index=q.row_id).loc[ids.row_id].to_numpy()
    assert np.isfinite(answer).all() and (answer>=0).all()
    with (a.output/'submission_14.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['row_id','sub_temp','sub_ec'])
        w.writerows((rid,'',format(float(v),'.17g')) for rid,v in zip(ids.row_id,answer))
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'torch':torch.__version__,'tabpfn':tabpfn.__version__}
    save(a.output/'manifest.json',{'status':'PASS','rows':1440,'temperature':'ALL_BLANK_BY_USER_REQUEST','training_rows':9600,'training_days':400,'consumed_lock_included_in_final_fit':True,'consumed_lock_rescored':False,'recipe':'season_v2_plus_DP1_ops_in_R3_plus_SG2_post','features':{'FULL_PFN':FULL,'FULL_R3':FULL_R3,'BASE_R3':BASE_R3},'r3_seeds':SEEDS,'pfn_contexts':[1,2,3,4],'season_notes':notes,'checks':check|sg2check|{'query_separability_prediction_checks':invariance},'code_sha256':{n:sha(HERE/n) for n in ['model.py','season.py','sg2post.py']},'inputs_sha256':{n:sha(a.data/n) for n in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']},'checkpoint_sha256':sha(a.checkpoint),'csv_sha256':sha(a.output/'submission_14.csv'),'sg2':{'reference':'all 400 labelled training days of F13/F47','evaluation_inputs':'same farm, hours 0..h of the row and earlier records only'},'versions':versions,'platform_submission':False})
    print('submission_14.csv ready: '+sha(a.output/'submission_14.csv'),flush=True)
if __name__=='__main__':main()
