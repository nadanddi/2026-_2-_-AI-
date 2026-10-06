from pathlib import Path
import sys,os,json,hashlib,gc,time
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'연구실/코덱스/analysis/ec_resolution_sequence_20261006_v1'));import stage1_v2 as S
sys.path.insert(0,str(H));from leaf_core_v1 import leaf_weights,weight_shapley
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
L=ROOT/'연구실/코덱스/local'/H.name
PAIRS=[('F47',160,161),('F13',98,112)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def prepare():
    orig,jobs,_=S.prepare();sets=[]
    for seed in [7,101,2024]:
        t,q=jobs['common_intersection'];sets.append((f'common_{seed}',t,q,seed,S.O/f'common_intersection_r3_{seed}.npz','raw_et'))
    t,q=jobs['trainfold1_purged'];sets.append(('purged1_7',t,q,7,S.O/'trainfold1_purged_r3_7.npz','raw_et'))
    for fold,targets in [(0,[('F47',160),('F13',112)]),(1,[('F47',161)])]:
        t,q=orig[fold];q=q[[(r.farm,int(r.day)) in targets for r in q.itertuples()]].reset_index(drop=True)
        assert len(q)==24*len(targets);sets.append((f'actual{fold}_7',t,q,7,S.C/f'DIAG10_{fold}_r3_7.npz','raw_et'))
    records=[]
    for name,t,q,seed,path,key in sets:
        assert not set(t.row_id)&set(q.row_id)
        with np.load(path,allow_pickle=False) as c:
            assert c['train_row_id'].tolist()==t.row_id.tolist()
            ix=pd.Index(c['row_id']).get_indexer(q.row_id);assert min(ix)>=0
        records.append(dict(name=name,seed=seed,train_rows=len(t),query_rows=len(q),train_hash=S.fhash(t,['row_id','sub_ec']+S.M.FULL),query_hash=S.fhash(q,['row_id','sub_ec']+S.M.FULL),cache_sha=sha(path),cache_path=str(path)))
    prep=dict(status='PREPARED_ET_SUPPORT_TRACE',jobs=records,source_sha=sha(__file__),core_sha=sha(H/'leaf_core_v1.py'),protocol_sha=sha(H/'PROTOCOL_v1.md'),original_preparation_sha=sha(S.H/'preparation_v2.json'),full_columns=S.M.FULL,fit=0,adoption=False)
    p=H/'preparation_v2.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==prep
    else:write(p,prep)
    return sets,prep
def profiles(name,t,q,w,pred,X):
    ty=t.sub_ec.to_numpy(float);ym=t.groupby(['farm','day']).sub_ec.transform('mean').to_numpy();out=[];rows=[];dayrows=[]
    for (farm,day),indices in q.groupby(['farm','day']).indices.items():
        indices=np.asarray(indices);assert len(indices)==24
        h0=indices[q.hour.iloc[indices].to_numpy()==0];assert len(h0)==1
        for level,v,truth in [('hour0',w[h0[0]],float(q.sub_ec.iloc[h0[0]])),('daymean',w[indices].mean(axis=0),float(q.sub_ec.iloc[indices].mean()))]:
            select=v>0;day_support=t.loc[select,['farm','day']].drop_duplicates()
            out.append(dict(context=name,farm=farm,day=int(day),level=level,prediction=float(v@ty),true_ec=truth,bias=float(v@ty-truth),high_row_weight=float(v[ty>=1].sum()),high_day_weight=float(v[ym>=1].sum()),F13_weight=float(v[t.farm.eq('F13')].sum()),F47_weight=float(v[t.farm.eq('F47')].sum()),support_rows=int(select.sum()),support_days=len(day_support),train_global_mean=float(ty.mean()),weighted_co2=float(v@X[:,S.M.FULL.index('in_co2')]),weighted_heating=float(v@X[:,S.M.FULL.index('act_heating')]),weighted_co2_h0=float(v@X[:,S.M.FULL.index('in_co2_h0')]),weighted_heating_h0=float(v@X[:,S.M.FULL.index('act_heating_h0')]),co2_missing_weight=float(v[t.in_co2.isna()].sum()),heating_missing_weight=float(v[t.act_heating.isna()].sum()),weighted_training_day_ec=float(v@ym)))
            g=t[['row_id','farm','day','hour','sub_ec','in_co2','in_co2_h0','act_heating','act_heating_h0','in_temp','in_hum','season']].copy();g['weight']=v;g['contribution']=v*ty;g=g[select];g=g.rename(columns={'farm':'train_farm','day':'train_day','hour':'train_hour'})
            g['context']=name;g['query_farm']=farm;g['query_day']=int(day);g['level']=level;rows.append(g)
            d=g.groupby(['train_farm','train_day'],as_index=False)[['weight','contribution']].sum();d['weighted_ec']=d.contribution/d.weight;d['context']=name;d['query_farm']=farm;d['query_day']=int(day);d['level']=level;dayrows.append(d)
    return out,pd.concat(rows,ignore_index=True),pd.concat(dayrows,ignore_index=True)
def first_split(name,forest,X,Q,t,q):
    out=[];y=t.sub_ec.to_numpy(float);ym=t.groupby(['farm','day']).sub_ec.transform('mean').to_numpy();ids={(r.farm,int(r.day),int(r.hour)):i for i,r in enumerate(q.itertuples())}
    for farm,good,bad in PAIRS:
        if (farm,good,0) not in ids or (farm,bad,0) not in ids:continue
        a,b=ids[farm,good,0],ids[farm,bad,0]
        for tree,est in enumerate(forest.estimators_):
            tr=est.tree_;node=0;mask=np.ones(len(X),bool)
            while tr.children_left[node]!=tr.children_right[node]:
                c=int(tr.feature[node]);th=float(tr.threshold[node]);ga=Q[a,c]<=th;ba=Q[b,c]<=th
                if ga!=ba:
                    assert mask.sum()==tr.n_node_samples[node]
                    for side,branch in [('good',ga),('bad',ba)]:
                        sel=mask&((X[:,c]<=th)==branch);child=int(tr.children_left[node] if branch else tr.children_right[node]);assert sel.sum()==tr.n_node_samples[child]
                        avg=float(y[sel].mean());assert abs(avg-float(tr.value[child,0,0]))<1e-10
                        out.append(dict(context=name,farm=farm,good=good,bad=bad,tree=tree,parent=node,child=child,feature=S.M.FULL[c],threshold=th,good_value=float(Q[a,c]),bad_value=float(Q[b,c]),side=side,rows=int(sel.sum()),days=len(t.loc[sel,['farm','day']].drop_duplicates()),parent_rows=int(mask.sum()),mean_ec=avg,high_row_fraction=float((y[sel]>=1).mean()),high_day_fraction=float((ym[sel]>=1).mean()),F13_fraction=float(t.farm.iloc[np.flatnonzero(sel)].eq('F13').mean())))
                    break
                mask&=((X[:,c]<=th)==ga);node=int(tr.children_left[node] if ga else tr.children_right[node])
    return out
def coalitions(farm,q):
    _,good,bad=next(p for p in PAIRS if p[0]==farm);a=q[(q.farm==farm)&(q.day==good)&(q.hour==0)].iloc[0];b=q[(q.farm==farm)&(q.day==bad)&(q.hour==0)].iloc[0]
    names=['season']+S.M.RAW;groups={n:[c for c in S.M.FULL if c==n or c.startswith(n+'_')] for n in names}
    changing=[n for n in names if groups[n] and not np.array_equal(a[groups[n]].to_numpy(float),b[groups[n]].to_numpy(float))]
    assert len(changing)==5
    frames=[]
    for mask in range(1<<len(changing)):
        row=a[S.M.FULL].copy()
        for i,g in enumerate(changing):
            if mask&(1<<i):row[groups[g]]=b[groups[g]]
        frames.append(row)
    return pd.DataFrame(frames,columns=S.M.FULL).astype(float),changing
def main():
    sets,prep=prepare()
    if '--prepare' in sys.argv:print('PREPARED_FIT0',flush=True);return
    L.mkdir(parents=True,exist_ok=True);lock=L/'worker.lock';write(lock,dict(pid=os.getpid(),source_sha=sha(__file__)))
    D=H/'results_v2';D.mkdir(exist_ok=False);profiles_all=[];support=[];support_day=[];splits=[];changes=[];phis=[];receipts=[]
    for name,t,q,seed,cache,key in sets:
        started=time.monotonic();model=S.M.et(seed)
        with threadpool_limits(limits=2):model.fit(t[S.M.FULL],t.sub_ec.to_numpy(float));pred=model.predict(q[S.M.FULL])
        im,forest=model.steps[0][1],model.steps[-1][1]
        assert forest.bootstrap is False and forest.criterion=='squared_error' and len(forest.estimators_)==600
        X=np.asarray(im.transform(t[S.M.FULL]),np.float32);Q=np.asarray(im.transform(q[S.M.FULL]),np.float32);assert X.shape[1]==38
        with np.load(cache,allow_pickle=False) as c:
            ix=pd.Index(c['row_id']).get_indexer(q.row_id);gap=float(np.max(abs(pred-c[key][ix])));assert gap<=1e-10
        with threadpool_limits(limits=2):tl=forest.apply(X).astype(np.int32);ql=forest.apply(Q).astype(np.int32)
        w=leaf_weights(tl,ql);ty=t.sub_ec.to_numpy(float);err=float(np.max(abs(w@ty-pred)));assert err<1e-10
        for j,est in enumerate(forest.estimators_):
            tr=est.tree_;leaf,count=np.unique(tl[:,j],return_counts=True)
            assert np.array_equal(tr.n_node_samples[leaf],count);assert np.array_equal(tr.weighted_n_node_samples[leaf],count)
        pp,rr,dd=profiles(name,t,q,w,pred,X);profiles_all.extend(pp);support.append(rr);support_day.append(dd);splits.extend(first_split(name,forest,X,Q,t,q))
        ids={(r.farm,int(r.day),int(r.hour)):i for i,r in enumerate(q.itertuples())}
        for farm,good,bad in PAIRS:
            if (farm,good,0) not in ids or (farm,bad,0) not in ids:continue
            for level in ['hour0','daymean']:
                wg=w[ids[farm,good,0]] if level=='hour0' else w[(q.farm==farm)&(q.day==good)].mean(axis=0)
                wb=w[ids[farm,bad,0]] if level=='hour0' else w[(q.farm==farm)&(q.day==bad)].mean(axis=0);delta=wb-wg;assert abs(delta.sum())<1e-12
                g=t[['row_id','farm','day','sub_ec']].copy();g['delta_weight']=delta;g['raw_contribution']=delta*ty;g['centered_contribution']=delta*(ty-ty.mean());g['context']=name;g['query_farm']=farm;g['level']=level;changes.append(g[g.delta_weight!=0])
        packed=dict(train_X=X,query_X=Q,train_y=ty,query_y=q.sub_ec.to_numpy(float),train_row_id=t.row_id.to_numpy(str),query_row_id=q.row_id.to_numpy(str),train_leaf=tl,query_leaf=ql,weights=w,raw_et=pred,imputer_median=im.statistics_)
        node_counts=[e.tree_.node_count for e in forest.estimators_];packed['offsets']=np.r_[0,np.cumsum(node_counts)]
        for field,attr in [('left','children_left'),('right','children_right'),('feature','feature'),('threshold','threshold'),('n_samples','n_node_samples'),('weighted_n_samples','weighted_n_node_samples')]:
            packed[field]=np.concatenate([getattr(e.tree_,attr) for e in forest.estimators_]);
            if field in ['left','right','feature','n_samples']:packed[field]=packed[field].astype(np.int32)
        packed['value']=np.concatenate([e.tree_.value[:,0,0] for e in forest.estimators_])
        if name=='common_7':
            for farm in ['F47','F13']:
                frame,names=coalitions(farm,q);CX=np.asarray(im.transform(frame),np.float32);cl=forest.apply(CX).astype(np.int32);cw=leaf_weights(tl,cl);phi=weight_shapley(cw,len(names));val=phi@ty
                old=pd.read_csv(S.H/'stage1_explanation_v1/grouped_contributions.csv');old=old[(old.model=='ET')&(old.farm==farm)].set_index('group')
                assert np.max(abs(val-np.array([old.loc[n,'prediction_change'] for n in names])))<1e-10
                for i,n in enumerate(names):
                    g=t[['row_id','farm','day','sub_ec']].copy();g['phi_weight']=phi[i];g['raw_contribution']=phi[i]*ty;g['centered_contribution']=phi[i]*(ty-ty.mean());g['query_farm']=farm;g['group']=n;phis.append(g[g.phi_weight!=0])
                for suffix,array in [('X',CX),('leaf',cl),('weights',cw),('phi',phi)]:packed[f'coalition_{farm}_{suffix}']=array
                packed[f'coalition_{farm}_groups']=np.asarray(names,str)
        dest=L/(name+'.npz');assert not dest.exists();np.savez_compressed(dest,**packed)
        meta=dict(name=name,seed=seed,npz_sha=sha(dest),train_hash=prep['jobs'][len(receipts)]['train_hash'],query_hash=prep['jobs'][len(receipts)]['query_hash'],replay_maxdiff=gap,weighted_prediction_maxdiff=err,nodes=int(packed['offsets'][-1]),seconds=time.monotonic()-started)
        write(dest.with_suffix('.json'),meta);receipts.append(meta);print('TRACE_COMPLETE',name,'replay',gap,'weights',err,'seconds',round(meta['seconds'],1),flush=True)
        del model,forest,packed,w,tl,ql;gc.collect()
    pd.DataFrame(profiles_all).to_csv(D/'support_profiles.csv',index=False);pd.concat(support,ignore_index=True).to_csv(D/'support_rows.csv',index=False);pd.concat(support_day,ignore_index=True).to_csv(D/'support_days.csv',index=False);pd.DataFrame(splits).to_csv(D/'first_split_training.csv',index=False);pd.concat(changes,ignore_index=True).to_csv(D/'pair_weight_changes.csv',index=False);pd.concat(phis,ignore_index=True).to_csv(D/'weight_shapley_rows.csv',index=False)
    write(D/'completion.json',dict(status='COMPLETE_ET_LEARNING_SUPPORT_TRACE',fits=6,models=receipts,source_sha=sha(__file__),preparation_sha=sha(H/'preparation_v2.json'),files={p.name:sha(p) for p in D.glob('*.csv')},adoption=False,PFN_traced=False,physical_causality=False))
    owner=json.loads(lock.read_text());assert owner['pid']==os.getpid();lock.unlink();print('TRACE_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()

