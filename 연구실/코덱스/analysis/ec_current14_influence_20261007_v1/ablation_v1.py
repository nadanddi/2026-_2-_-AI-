from pathlib import Path
import sys,os,json,time,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));import runner_v4 as R
sys.path.insert(0,str(R.ROOT/'연구실/코덱스/analysis/ec_training_trace_20261006_v1'))
from leaf_core_v1 import leaf_weights
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
ARMS={'D1':[139],'D2':[139,231]}
def trace(model,t,q,arm):
    im,forest=model.steps[0][1],model.steps[-1][1]
    X=np.asarray(im.transform(t[R.M.FULL_R3]),np.float32);Q=np.asarray(im.transform(q[R.M.FULL_R3]),np.float32)
    tl=forest.apply(X).astype(np.int32);ql=forest.apply(Q).astype(np.int32);w=leaf_weights(tl,ql)
    pred=forest.predict(Q);assert np.max(abs(w@t.sub_ec.to_numpy(float)-pred))<1e-10
    packed=dict(train_X=X,query_X=Q,train_y=t.sub_ec.to_numpy(float),query_y=q.sub_ec.to_numpy(float),train_row_id=t.row_id.to_numpy(str),query_row_id=q.row_id.to_numpy(str),train_leaf=tl,query_leaf=ql,weights=w,raw_et=pred,imputer_median=im.statistics_)
    packed['offsets']=np.r_[0,np.cumsum([e.tree_.node_count for e in forest.estimators_])]
    for field,attr in [('left','children_left'),('right','children_right'),('feature','feature'),('threshold','threshold'),('n_samples','n_node_samples'),('weighted_n_samples','weighted_n_node_samples')]:
        packed[field]=np.concatenate([getattr(e.tree_,attr) for e in forest.estimators_])
        if field in ['left','right','feature','n_samples']:packed[field]=packed[field].astype(np.int32)
    packed['value']=np.concatenate([e.tree_.value[:,0,0] for e in forest.estimators_])
    O=R.L/'trace';O.mkdir(exist_ok=True);dest=O/(arm+'.npz');assert not dest.exists();np.savez_compressed(dest,**packed)
    R.write(dest.with_suffix('.json'),dict(arm=arm,seed=7,fold=1,sha=R.sha(dest),nodes=int(packed['offsets'][-1]),source_sha=R.sha(__file__)))
    records=[];supports=[];ym=t.groupby(['farm','day']).sub_ec.transform('mean').to_numpy(float)
    smooth_w=np.stack([.5*w[i]+.5*w[:i+1].mean(axis=0) for i in range(len(q))])
    for level,v in [('hour0',w[0]),('raw_day',w.mean(axis=0)),('smooth_day',smooth_w.mean(axis=0))]:
        assert abs(v.sum()-1)<1e-12
        records.append(dict(arm=arm,level=level,prediction=float(v@t.sub_ec.to_numpy(float)),truth=float(q.sub_ec.iloc[0] if level=='hour0' else q.sub_ec.mean()),high_day_weight=float(v[ym>=1].sum()),high_row_weight=float(v[t.sub_ec.to_numpy()>=1].sum())))
        d=t[['farm','day']].copy();d['weight']=v;d['contribution']=v*t.sub_ec.to_numpy(float);d=d.groupby(['farm','day'],as_index=False)[['weight','contribution']].sum();d=d[d.weight>0];d['weighted_ec']=d.contribution/d.weight;d['arm']=arm;d['level']=level;supports.append(d)
    return records,pd.concat(supports,ignore_index=True)
def main():
    raw,jobs,prep=R.prepare();receipt=json.loads((H/'baseline_receipt_v4.json').read_text(encoding='utf-8'));assert receipt['rows_sha']==R.sha(R.L/'baseline_rows.csv')
    registration=dict(source_sha=R.sha(__file__),baseline_receipt_sha=R.sha(H/'baseline_receipt_v4.json'),preparation_sha=R.sha(H/'preparation_v4.json'),arms=ARMS,ET_only=True,additional_baseline_trace_fit=1,max_deletion_fits=60,fit=0)
    pp=H/'ablation_preparation_v1.json'
    if pp.exists():assert json.loads(pp.read_text(encoding='utf-8'))==registration
    else:R.write(pp,registration)
    if '--prepare' in sys.argv:print('ABLATION_PREPARED_FIT0',flush=True);return
    R.write(R.L/'ablation.lock',dict(pid=os.getpid(),source_sha=R.sha(__file__)))
    frames=[];meta=[];profiles=[];support=[];fits=0
    for k,(t,q,pfn) in jobs.items():
        base={s:R.cache_r3(k,s,t,q) for s in R.SEEDS}
        ref=set(t[['farm','day']].itertuples(index=False,name=None));ec=t.groupby(['farm','day']).sub_ec.mean();structure=R.SG.prepare(raw,ref);cal=R.SG.ref_calendar(structure,ref)
        if k==1:
            qt=q[(q.farm=='F47')&(q.day==161)].sort_values('hour').reset_index(drop=True);assert len(qt)==24
            model=R.M.et(7)
            with threadpool_limits(limits=2):model.fit(t[R.M.FULL_R3],t.sub_ec.to_numpy(float));model.steps[-1][1].n_jobs=1;rp=model.predict(q[R.M.FULL_R3])
            assert np.max(abs(rp-base[7]['et']))<1e-10
            a,b=trace(model,t,qt,'BASE');profiles.extend(a);support.append(b);del model;gc.collect()
        for arm,days in ARMS.items():
            removed=t.farm.eq('F47')&t.day.isin(days);tt=t.loc[~removed].reset_index(drop=True);ets=[];r3=[]
            for s in R.SEEDS:
                start=time.monotonic();out=R.L/'ablation';out.mkdir(exist_ok=True);dest=out/f'{arm}_{k}_{s}.npz';assert not dest.exists()
                if removed.any():
                    model=R.M.et(s)
                    with threadpool_limits(limits=2):
                        model.fit(tt[R.M.FULL_R3],tt.sub_ec.to_numpy(float));model.steps[-1][1].n_jobs=1;et=model.predict(q[R.M.FULL_R3])
                    fits+=1
                    if k==1 and s==7:
                        a,b=trace(model,tt,qt,arm);profiles.extend(a);support.append(b)
                    del model;gc.collect()
                else:et=base[s]['et'].copy();assert np.array_equal(et,base[s]['et'])
                assert np.isfinite(et).all();ets.append(et);np.savez_compressed(dest,row_id=q.row_id.to_numpy(str),train_row_id=tt.row_id.to_numpy(str),removed_row_id=t.loc[removed,'row_id'].to_numpy(str),et=et)
                m=dict(arm=arm,k=k,seed=s,sha=R.sha(dest),prep_sha=R.sha(pp),removed_rows=int(removed.sum()),new_fit=bool(removed.any()),train_hash=R.S.fhash(tt,['row_id','sub_ec']+R.M.FULL_R3),seconds=time.monotonic()-start);R.write(dest.with_suffix('.json'),m);meta.append(m)
                mix=.6*et+.3*base[s]['lgb']+.1*base[s]['mlp'];r3.append(mix);f=R.post(k,arm,s,t,q,.8*mix+.2*pfn,structure,cal,ec,ref);f['raw_et']=et;frames.append(f)
                print('ET_DELETION_COMPLETE',arm,k,s,'removed',int(removed.sum()),'seconds',round(time.monotonic()-start,1),flush=True)
            f=R.post(k,arm,'ensemble',t,q,.8*np.mean(r3,axis=0)+.2*pfn,structure,cal,ec,ref);f['raw_et']=np.mean(ets,axis=0);frames.append(f)
    rows=R.L/'ablation_rows.csv';assert not rows.exists();pd.concat(frames,ignore_index=True).to_csv(rows,index=False)
    pd.DataFrame(profiles).to_csv(H/'influence_support_profiles_v1.csv',index=False);pd.concat(support,ignore_index=True).to_csv(H/'influence_support_days_v1.csv',index=False)
    R.write(H/'ablation_receipt_v1.json',dict(status='COMPLETE_ET_ONLY_DELETION',rows=69120,rows_sha=R.sha(rows),models=meta,new_ET_fits=fits,additional_baseline_trace_fit=1,source_sha=R.sha(__file__),preparation_sha=R.sha(pp),adoption=False))
    assert json.loads((R.L/'ablation.lock').read_text())['pid']==os.getpid();(R.L/'ablation.lock').unlink();print('ABLATION_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()
