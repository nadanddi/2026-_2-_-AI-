"""Create cleaned datasets, fold-safe CV, and fit a fully cleaned-data R3 model."""
from pathlib import Path
import os,sys,json,hashlib,time,gc,argparse,platform
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'집/코덱스/local'/H.name
import recipe_ec_v1 as C
import numpy as np,pandas as pd,joblib,sklearn,lightgbm
from threadpoolctl import threadpool_limits
OLD=ROOT/'연구실/코덱스/analysis/ec_current14_influence_20261007_v1/preparation_v4.json'
BASE=ROOT/'연구실/코덱스/local/ec_current14_influence_20261007_v1/base'
OOF=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv'
DATA=Path(C.env.DATA);SEEDS=C.SEEDS;ALPHA=.025/2
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def csv_new(p,f):assert not p.exists(),p;f.to_csv(p,index=False)
def fhash(f,cols):return hashlib.sha256(pd.util.hash_pandas_object(f[cols],index=False).values.tobytes()).hexdigest()
def ordered(f,ids):return f.set_index('row_id').loc[list(ids)].reset_index()
def pins():
    paths=[Path(__file__),H/'PLAN_v1.md',H/'recipe_ec_v1.py',H/'season_transform_v1.py',H/'predict_model_v1.py',Path(C.env.__file__),OLD,OOF,DATA/'train_X.csv',DATA/'train_y.csv']
    return {str(p):sha(p) for p in paths}
def checkpins():
    reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'))
    for p,v in reg['pins'].items():assert sha(p)==v,p
    for p,v in reg['cachepins'].items():assert sha(p)==v,p
    return reg
def load_public():
    assert sha(DATA/'train_X.csv')=='21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300'
    z=pd.read_csv(OOF,usecols=['validator','row_id','y'],float_precision='round_trip');z=z[z.validator=='DIAG10'];y=z[['row_id','y']].drop_duplicates().rename(columns={'y':'sub_ec'})
    assert len(y)==8640 and y.row_id.is_unique and np.isfinite(y.sub_ec).all() and (y.sub_ec>=0).all()
    x=pd.read_csv(DATA/'train_X.csv');x=x[x.row_id.isin(y.row_id)].reset_index(drop=True)
    assert x.row_id.is_unique and set(x.row_id)==set(y.row_id)
    f=C.features(x[['row_id']+C.RAW]).merge(y,on='row_id',validate='one_to_one');return x,y,f

def profile(x,y):
    a=C.identify(x);assert x.row_id.is_unique and y.row_id.is_unique and set(x.row_id)==set(y.row_id)
    sizes=a.groupby(['farm','day']).size();assert (sizes==24).all()
    assert all(set(g.hour)==set(range(24)) for _,g in a.groupby(['farm','day']))
    assert np.isfinite(y.sub_ec).all() and (y.sub_ec>=0).all()
    cols={}
    for name in x:
        s=x[name];v=s.dropna();info=dict(dtype=str(s.dtype),missing=int(s.isna().sum()),unique=int(s.nunique()))
        if pd.api.types.is_numeric_dtype(s):info.update(min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None,quantile={str(q):float(s.quantile(q)) if len(v) else None for q in (0,.01,.25,.5,.75,.99,1)},infinite=int(np.isinf(v).sum()))
        cols[name]=info
    return dict(rows=len(x),columns=len(x.columns),days=len(sizes),farms=a.farm.value_counts().to_dict(),dtype_missing_ranges=cols,target_min=float(y.sub_ec.min()),target_max=float(y.sub_ec.max()),target_negative=int((y.sub_ec<0).sum()),exact_duplicate_rows=int(x.duplicated().sum()),duplicate_keys=int(x.row_id.duplicated().sum()),xy_same_keys=True,all_24_hours=True,coverage={farm:dict(first=int(g.day.min()),last=int(g.day.max()),observed_days=int(g.day.nunique()),gaps=sorted(set(range(int(g.day.min()),int(g.day.max())+1))-set(g.day))) for farm,g in a.groupby('farm')})

def select(t,q):
    d=t.groupby(['farm','day'],sort=True)[C.FULL_R3].mean();ym=t.groupby(['farm','day'],sort=True).sub_ec.mean().reindex(d.index)
    arr=d.to_numpy(float);med=np.nanmedian(arr,axis=0);assert np.isfinite(med).all();filled=np.where(np.isnan(arr),med,arr);mu=filled.mean(0);sd=filled.std(0);sd[sd<1e-12]=1;z=(filled-mu)/sd
    idx=list(d.index);assert (t.groupby(['farm','day']).size()==24).all()
    qd=q.groupby(['farm','day'],sort=True)[C.FULL_R3].mean();qy=q.groupby(['farm','day'],sort=True).sub_ec.mean().reindex(qd.index)
    qarr=qd.to_numpy(float);qz=(np.where(np.isnan(qarr),med,qarr)-mu)/sd
    def classify(keys,points,target,scope):
        removed=set();detail=[]
        for i,(farm,day) in enumerate(keys):
            eligible=np.array([ff==farm and abs(dd-day)>1 for ff,dd in idx]);assert eligible.sum()>=5
            distance=np.sqrt(np.mean((z-points[i])**2,axis=1));near=sorted(np.flatnonzero(eligible),key=lambda j:(distance[j],idx[j]))[:5]
            neigh=float(np.median(ym.iloc[near]));level=float(target.iloc[i]);bad=bool(level>=1 and level-neigh>.5)
            if bad:removed.add((str(farm),int(day)))
            detail.append(dict(scope=scope,farm=str(farm),day=int(day),y=level,neighbor_y=neigh,gap=level-neigh,remove=bad,neighbors=[dict(farm=str(idx[j][0]),day=int(idx[j][1]),y=float(ym.iloc[j]),distance=float(distance[j])) for j in near]))
        return removed,detail
    rt,dt=classify(idx,z,ym,'train');rq,dq=classify(list(qd.index),qz,qy,'query_conditional_only')
    snap=dict(train_keys=np.asarray([f'{f}_{d:03d}' for f,d in idx]),query_keys=np.asarray([f'{f}_{d:03d}' for f,d in qd.index]),train_z=z,query_z=qz,train_y=ym.to_numpy(),query_y=qy.to_numpy(),median=med,mean=mu,std=sd)
    return rt,rq,dt+dq,snap

def dataset(x,y,removed,dest):
    dest.mkdir(parents=True,exist_ok=True);a=C.identify(x);mask=np.array([(f,int(d)) in removed for f,d in zip(a.farm,a.day)])
    cx=x.loc[~mask].reset_index(drop=True);cy=ordered(y,cx.row_id);assert len(x)-len(cx)==24*len(removed)
    csv_new(dest/'train_X_clean_v1.csv',cx);csv_new(dest/'train_y_clean_v1.csv',cy)
    csv_new(dest/'removed_row_ids_v1.csv',x.loc[mask,['row_id']]);csv_new(dest/'removed_days_v1.csv',pd.DataFrame(sorted(removed),columns=['farm','day']))
    m=dict(before=profile(x,y),after=profile(cx,cy),removed_days=len(removed),removed_rows=int(mask.sum()),preserved_original_row_ids=True,source_rows_not_renumbered=True,cleaning_authority='user explicitly requested full-day deletion by fixed proxy',hashes={p.name:sha(p) for p in dest.glob('*.csv')})
    save(dest/'cleaning_manifest_v1.json',m);return cx,cy,m

def build_jobs(x,y,f,write_artifacts=False):
    old=json.loads(OLD.read_text(encoding='utf8'));jobs={};recs=[];cachepins={}
    for rec in old['records']:
        k=rec['k'];rt=ordered(x,rec['train_ids']);rq=ordered(x,rec['query_ids']);t=ordered(f,rec['train_ids']);q=ordered(f,rec['query_ids']);t,q,n=C.season(t,q,rt)
        assert fhash(t,['row_id','sub_ec']+C.FULL_R3)==rec['train_full_hash'] and fhash(q,['row_id','sub_ec']+C.FULL_R3)==rec['query_full_hash']
        assert not set(t.row_id)&set(q.row_id)
        for farm,day in q[['farm','day']].drop_duplicates().itertuples(index=False,name=None):assert abs(t.loc[t.farm==farm,'day']-day).min()>=2
        removals,removed_q,detail,snap=select(t,q)
        kept=[rid for rid,farm,day in t[['row_id','farm','day']].itertuples(index=False,name=None) if (farm,int(day)) not in removals]
        rx=ordered(rt,kept);cy=ordered(y,kept);ct=C.features(rx[['row_id']+C.RAW]).merge(cy,on='row_id',validate='one_to_one');ct=ordered(ct,kept)
        cq=C.features(rq[['row_id']+C.RAW]).merge(ordered(y,rq.row_id),on='row_id',validate='one_to_one');cq=ordered(cq,q.row_id);ct,cq,cn=C.season(ct,cq,rx)
        assert set(ct.row_id)==set(kept) and all((farm,int(day)) not in removals for farm,day in ct[['farm','day']].itertuples(index=False,name=None))
        base={}
        for seed in SEEDS:
            p=BASE/f'{k}_{seed}.npz';mp=p.with_suffix('.json');meta=json.loads(mp.read_text(encoding='utf8'));assert meta['sha']==sha(p) and meta['prep_sha']==sha(OLD)
            with np.load(p,allow_pickle=False) as z:
                assert z['row_id'].tolist()==q.row_id.tolist() and z['train_row_id'].tolist()==t.row_id.tolist();base[seed]={kind:z[kind].copy() for kind in C.WEIGHTS}
            cachepins[str(p)]=sha(p);cachepins[str(mp)]=sha(mp)
        if write_artifacts:
            dest=L/'cv_data'/f'fold{k}';cx0,cy0,dm=dataset(rt,ordered(y,rt.row_id),removals,dest)
            csv_new(dest/'query_retained_ids_v1.csv',q.loc[[(ff,int(dd)) not in removed_q for ff,dd in zip(q.farm,q.day)],['row_id']])
            csv_new(dest/'query_removed_ids_v1.csv',q.loc[[(ff,int(dd)) in removed_q for ff,dd in zip(q.farm,q.day)],['row_id']])
            save(dest/'selection_detail_v1.json',detail);np.savez_compressed(dest/'selector_snapshot_v1.npz',**snap)
            save(dest/'clean_season_v1.json',cn)
        jobs[k]=(t,q,ct,cq,base,removals,removed_q,cn)
        recs.append(dict(k=k,original_train_ids=t.row_id.tolist(),clean_train_ids=ct.row_id.tolist(),query_ids=q.row_id.tolist(),removed_train_days=sorted(removals),removed_query_days=sorted(removed_q),train_original_hash=fhash(t,['row_id','sub_ec']+C.FULL_R3),query_original_hash=fhash(q,['row_id','sub_ec']+C.FULL_R3),clean_train_hash=fhash(ct,['row_id','sub_ec']+C.FULL_R3),clean_query_hash=fhash(cq,['row_id','sub_ec']+C.FULL_R3),clean_bounds=[float(ct.sub_ec.min()),float(ct.sub_ec.max())]))
    return jobs,recs,cachepins

def prepare():
    assert not (H/'registration_v1.json').exists();L.mkdir(parents=True,exist_ok=True)
    x,y,f=load_public();t,q,n=C.season(f,f.iloc[:0],x);removed,_,detail,snap=select(t,q)
    cx,cy,cleanlog=dataset(x,y,removed,L/'dataset_public');save(H/'profile_public_v1.json',cleanlog)
    save(L/'dataset_public/selection_detail_v1.json',detail);np.savez_compressed(L/'dataset_public/selector_snapshot_v1.npz',**snap)
    jobs,recs,cp=build_jobs(x,y,f,True)
    reg=dict(status='REGISTERED_FIT0',pins=pins(),cachepins=cp,folds=recs,public_removed_days=sorted(removed),public_rows=8640,public_clean_rows=len(cx),seeds=list(SEEDS),weights=C.WEIGHTS,features=dict(et=C.FULL_R3,lgb=C.BASE_R3,mlp=C.BASE_R3),alpha=ALPHA,max_cv_candidate_models=90,baseline_replay_models=3,final_models=9,conditional_query_labels_used_only_for_scoring=True,global_clean_data_used_by_cv=False,full400_labels_loaded_only_after_cv=True,adoption=False,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__))
    save(H/'registration_v1.json',reg);print('PREPARED',len(cx),'rows removed',len(removed),'fold removals',[(r['k'],len(r['removed_train_days']),len(r['removed_query_days'])) for r in recs],flush=True)

def jobs_from_registry():
    reg=checkpins();x,y,f=load_public();jobs,recs,cp=build_jobs(x,y,f,False)
    assert json.loads(json.dumps(recs))==reg['folds'] and cp==reg['cachepins'];return jobs

def fit_predict(kind,seed,t,q):
    model=C.factory(kind,seed);cols=C.feature_columns(kind);start=time.monotonic()
    with threadpool_limits(limits=2):
        model.fit(t[cols],t.sub_ec.to_numpy(float))
        if kind=='et':model.steps[-1][1].n_jobs=1
        p=np.asarray(model.predict(q[cols]),float);tp=np.asarray(model.predict(t[cols]),float)
        assert np.max(abs(p[:8]-np.asarray(model.predict(q.iloc[:8][cols]),float)))<1e-10
    receipt=dict(kind=kind,seed=seed,rows=len(t),train_ids_sha=hashlib.sha256('\n'.join(t.row_id).encode()).hexdigest(),train_feature_hash=fhash(t,['row_id','sub_ec']+cols),seconds=time.monotonic()-start)
    return model,p,tp,receipt

def run(jobs,ks):
    assert (H/'critique_plan_v1.md').exists();D=L/'cv_predictions';D.mkdir(exist_ok=True)
    for k in ks:
        checkpins();dest=D/f'fold{k}.csv';mp=dest.with_suffix('.json')
        if dest.exists() or mp.exists():assert dest.exists() and mp.exists() and json.loads(mp.read_text(encoding='utf8'))['sha']==sha(dest);continue
        t,q,ct,cq,base,removed,rq,notes=jobs[k]
        if k==0 and not (H/'baseline_replay_v1.json').exists():
            gaps={}
            for kind in C.WEIGHTS:
                model,p,tp,receipt=fit_predict(kind,7,t,q);gap=float(np.max(abs(p-base[7][kind])));assert gap<(1e-8 if kind=='mlp' else 1e-10),(kind,gap);gaps[kind]=gap;del model;gc.collect()
            save(H/'baseline_replay_v1.json',dict(gaps=gaps,models=3,registration=sha(H/'registration_v1.json')))
        frames=[];fitinfo=[];trainraw=[];candidates=[]
        for seed in SEEDS:
            preds={};tps={}
            for kind in C.WEIGHTS:
                model,p,tp,receipt=fit_predict(kind,seed,ct,cq);preds[kind]=p;tps[kind]=tp;fitinfo.append(receipt);del model;gc.collect()
                print('FIT',k,seed,kind,'train_days',len(ct)//24,'seconds',round(receipt['seconds'],1),flush=True)
            candidates.append(sum(C.WEIGHTS[n]*preds[n] for n in C.WEIGHTS));trainraw.append(sum(C.WEIGHTS[n]*tps[n] for n in C.WEIGHTS))
            for arm,raw,tr,fr in [('BASE',sum(C.WEIGHTS[n]*base[seed][n] for n in C.WEIGHTS),t,q),('CLEAN',candidates[-1],ct,cq)]:
                frame=fr[['row_id','farm','day','hour','sub_ec']].copy();frame['k']=k;frame['arm']=arm;frame['seed']=str(seed);frame['raw_prediction']=raw;frame['prediction']=np.clip(C.shrink(raw,fr),tr.sub_ec.min(),tr.sub_ec.max());frame['query_removed']=[(ff,int(dd)) in rq for ff,dd in zip(fr.farm,fr.day)];frame['constant_prediction']=tr.sub_ec.mean();frames.append(frame)
        for arm,raw,tr,fr in [('BASE',np.mean([sum(C.WEIGHTS[n]*base[s][n] for n in C.WEIGHTS) for s in SEEDS],axis=0),t,q),('CLEAN',np.mean(candidates,axis=0),ct,cq)]:
            frame=fr[['row_id','farm','day','hour','sub_ec']].copy();frame['k']=k;frame['arm']=arm;frame['seed']='ensemble';frame['raw_prediction']=raw;frame['prediction']=np.clip(C.shrink(raw,fr),tr.sub_ec.min(),tr.sub_ec.max());frame['query_removed']=[(ff,int(dd)) in rq for ff,dd in zip(fr.farm,fr.day)];frame['constant_prediction']=tr.sub_ec.mean();frames.append(frame)
        trainmetrics={str(seed):float(np.sqrt(np.mean((np.clip(C.shrink(raw,ct),ct.sub_ec.min(),ct.sub_ec.max())-ct.sub_ec)**2))) for seed,raw in zip(SEEDS,trainraw)}
        trainmetrics['ensemble']=float(np.sqrt(np.mean((np.clip(C.shrink(np.mean(trainraw,axis=0),ct),ct.sub_ec.min(),ct.sub_ec.max())-ct.sub_ec)**2)))
        allf=pd.concat(frames,ignore_index=True);csv_new(dest,allf);save(mp,dict(k=k,sha=sha(dest),registration=sha(H/'registration_v1.json'),candidate_models=9,fitinfo=fitinfo,clean_train_rmse=trainmetrics,train_ids=ct.row_id.tolist(),clean_bounds=[float(ct.sub_ec.min()),float(ct.sub_ec.max())]));print('FOLD_COMPLETE',k,flush=True)

def score(ks,label):
    frames=[]
    for k in ks:
        p=L/'cv_predictions'/f'fold{k}.csv';m=json.loads(p.with_suffix('.json').read_text(encoding='utf8'));assert m['sha']==sha(p) and m['registration']==sha(H/'registration_v1.json');frames.append(pd.read_csv(p,float_precision='round_trip'))
    o=pd.concat(frames,ignore_index=True);means=o[o.arm=='BASE'].groupby(['farm','day']).sub_ec.mean();o['high']=[means[(f,d)]>=1 for f,d in zip(o.farm,o.day)]
    scopes={'all':np.ones(len(o),bool),'filtered':~o.query_removed,'removed':o.query_removed,'normal':~o.high,'high':o.high,'pass2_all':o.day>=179,'pass2_filtered':(o.day>=179)&~o.query_removed}
    for farm in ('F13','F47'):
        for pas,pm in [('pass1',o.day<179),('pass2',o.day>=179)]:
            scopes[f'{farm}_{pas}_all']=(o.farm==farm)&pm;scopes[f'{farm}_{pas}_filtered']=(o.farm==farm)&pm&~o.query_removed
    groups=[]
    for scope,mask in scopes.items():
        for (arm,seed),g in o.loc[mask].groupby(['arm','seed']):
            if len(g):groups.append(dict(scope=scope,arm=arm,seed=str(seed),rows=len(g),days=g.groupby(['farm','day']).ngroups,rmse=float(np.sqrt(np.mean((g.prediction-g.sub_ec)**2))),bias=float((g.prediction-g.sub_ec).mean()),constant_rmse=float(np.sqrt(np.mean((g.constant_prediction-g.sub_ec)**2)))))
    boots=[]
    for scope in ('filtered','all'):
        g=o.loc[scopes[scope]&o.seed.eq('ensemble')].copy();g['block']=g.day//5;g['loss']=(g.prediction-g.sub_ec)**2
        ss=g.groupby(['arm','farm','block']).loss.sum();nc=g[g.arm=='BASE'].groupby(['farm','block']).size();keys=list(nc.index);rng=np.random.default_rng(20261009);draw=[]
        for farm in ('F13','F47'):
            ids=np.array([i for i,v in enumerate(keys) if v[0]==farm]);assert len(ids);draw.append(ids[rng.integers(0,len(ids),(20000,len(ids)))])
        ix=np.concatenate(draw,1);a=ss.loc['CLEAN'].reindex(nc.index).to_numpy();b=ss.loc['BASE'].reindex(nc.index).to_numpy();n=nc.to_numpy();delta=(a-b)[ix].sum(1);dr=np.sqrt(a[ix].sum(1)/n[ix].sum(1))-np.sqrt(b[ix].sum(1)/n[ix].sum(1))
        boots.append(dict(scope=scope,p_worse=float((delta>=0).mean()),rmse_delta=float(np.sqrt(a.sum()/n.sum())-np.sqrt(b.sum()/n.sum())),ci95_delta=[float(v) for v in np.quantile(dr,[.025,.975])],alpha=ALPHA,blocks=len(keys)))
    folds=[]
    for (k,arm,seed),g in o.groupby(['k','arm','seed']):folds.append(dict(k=int(k),arm=arm,seed=str(seed),scope='all',rmse=float(np.sqrt(np.mean((g.prediction-g.sub_ec)**2)))))
    save(H/f'{label}_score_v1.json',dict(status='FULL_DIAG10' if len(ks)==10 else 'PARTIAL_DIAG10',folds=ks,groups=groups,bootstrap=boots,adoption=False,conditional_selection_warning='filtered query eligibility uses held-out daily EC; this population cannot be selected in actual evaluation'))
    csv_new(H/f'{label}_groups_v1.csv',pd.DataFrame(groups));csv_new(H/f'{label}_folds_v1.csv',pd.DataFrame(folds));print('SCORED',label,boots,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','first','rest','score','final_fit']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='first':jobs=jobs_from_registry();run(jobs,[0]);score([0],'midpoint')
    elif a.mode=='rest':assert (H/'critique_midpoint_v1.md').exists();jobs=jobs_from_registry();run(jobs,list(range(1,10)));score(list(range(10)),'final')
    elif a.mode=='score':checkpins();score(list(range(10)),'final')
    elif a.mode=='final_fit':raise RuntimeError('Final-fit implementation will be separately registered before CV starts.')
