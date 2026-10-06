from pathlib import Path
import sys,os,json,time,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OLD=ROOT/'연구실/코덱스/analysis/ec_current14_influence_20261007_v1'
sys.path.insert(0,str(OLD));import runner_v4 as B
sys.path.insert(0,str(H));from core_v1 import project_columns,DROP
sys.path.insert(0,str(ROOT/'연구실/코덱스/analysis/ec_training_trace_20261006_v1'));from leaf_core_v1 import leaf_weights
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
L=ROOT/'연구실/코덱스/local'/H.name;COLS=project_columns(B.M.FULL_R3);ARM='TWO_H0_DROP'
assert len(COLS)==45 and set(B.M.FULL_R3)-set(COLS)==set(DROP)
def load_base(k,s,t,q):
    path=B.L/'base'/f'{k}_{s}.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert meta['sha']==B.sha(path) and meta['prep_sha']==B.sha(OLD/'preparation_v4.json') and meta['k']==k and meta['seed']==s
    with np.load(path,allow_pickle=False) as c:
        assert c['row_id'].tolist()==q.row_id.tolist() and c['train_row_id'].tolist()==t.row_id.tolist()
        out={n:c[n].copy() for n in ['et','lgb','mlp']}
    assert all(v.shape==(len(q),) and np.isfinite(v).all() for v in out.values());return out
def prepare():
    raw,jobs,oldprep=B.prepare();done=json.loads((OLD/'완료상태_v1.json').read_text(encoding='utf-8'));assert done['status']=='COMPLETE_CURRENT14_AND_ET_DELETION_DIAGNOSTICS'
    oldreceipt=json.loads((OLD/'baseline_receipt_v4.json').read_text(encoding='utf-8'));assert oldreceipt['rows_sha']==B.sha(B.L/'baseline_rows.csv')
    records=[]
    for k,(t,q,pfn) in jobs.items():
        for s in B.SEEDS:load_base(k,s,t,q)
        records.append(dict(k=k,train_rows=len(t),query_rows=len(q),train_hash=B.S.fhash(t,['row_id','sub_ec']+COLS),query_hash=B.S.fhash(q,['row_id','sub_ec']+COLS),base_caches={str(s):B.sha(B.L/'base'/f'{k}_{s}.npz') for s in B.SEEDS}))
    parenttrace=B.L/'trace'/'BASE.npz';pt=json.loads(parenttrace.with_suffix('.json').read_text(encoding='utf-8'));assert pt['sha']==B.sha(parenttrace)
    info=dict(status='PREPARED_TWO_H0_DIAGNOSTIC_FIT0',source_sha=B.sha(__file__),core_sha=B.sha(H/'core_v1.py'),plan_sha=B.sha(H/'PLAN_v1.md'),addendum_sha=B.sha(H/'PLAN_ADDENDUM_v1.md'),old_preparation_sha=B.sha(OLD/'preparation_v4.json'),old_completion_sha=B.sha(OLD/'완료상태_v1.json'),old_receipt_sha=B.sha(OLD/'baseline_receipt_v4.json'),old_runner_sha=B.sha(OLD/'runner_v4.py'),parent_trace_sha=B.sha(parenttrace),columns=COLS,dropped=list(DROP),records=records,new_fit=0)
    p=H/'preparation_v1.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==info
    else:B.write(p,info)
    return raw,jobs,info
def save_trace(model,t,q):
    qt=q[(q.farm=='F47')&(q.day==161)].sort_values('hour').reset_index(drop=True);assert len(qt)==24
    im,forest=model.steps[0][1],model.steps[-1][1];assert not forest.bootstrap and forest.criterion=='squared_error'
    X=np.asarray(im.transform(t[COLS]),np.float32);Q=np.asarray(im.transform(qt[COLS]),np.float32);tl=forest.apply(X).astype(np.int32);ql=forest.apply(Q).astype(np.int32);w=leaf_weights(tl,ql);ty=t.sub_ec.to_numpy(float);pred=forest.predict(Q)
    assert np.max(abs(w@ty-pred))<1e-10
    packed=dict(train_X=X,query_X=Q,train_y=ty,query_y=qt.sub_ec.to_numpy(float),train_row_id=t.row_id.to_numpy(str),query_row_id=qt.row_id.to_numpy(str),train_leaf=tl,query_leaf=ql,weights=w,raw_et=pred,imputer_median=im.statistics_,columns=np.asarray(COLS,str))
    packed['offsets']=np.r_[0,np.cumsum([e.tree_.node_count for e in forest.estimators_])]
    for field,attr in [('left','children_left'),('right','children_right'),('feature','feature'),('threshold','threshold'),('n_samples','n_node_samples'),('weighted_n_samples','weighted_n_node_samples')]:
        packed[field]=np.concatenate([getattr(e.tree_,attr) for e in forest.estimators_])
        if field in ['left','right','feature','n_samples']:packed[field]=packed[field].astype(np.int32)
    packed['value']=np.concatenate([e.tree_.value[:,0,0] for e in forest.estimators_]);dest=L/'trace.npz';assert not dest.exists();np.savez_compressed(dest,**packed)
    B.write(dest.with_suffix('.json'),dict(arm=ARM,k=1,seed=7,sha=B.sha(dest),source_sha=B.sha(__file__),prep_sha=B.sha(H/'preparation_v1.json'),nodes=int(packed['offsets'][-1]),columns=COLS))
    ym=t.groupby(['farm','day']).sub_ec.transform('mean').to_numpy(float);sw=np.array([.5*w[i]+.5*w[:i+1].mean(axis=0) for i in range(24)]);profiles=[];supports=[]
    for level,v in [('hour0',w[0]),('raw_day',w.mean(axis=0)),('smooth_day',sw.mean(axis=0))]:
        profiles.append(dict(arm=ARM,level=level,prediction=float(v@ty),truth=float(qt.sub_ec.iloc[0] if level=='hour0' else qt.sub_ec.mean()),high_day_weight=float(v[ym>=1].sum()),high_row_weight=float(v[ty>=1].sum())))
        d=t[['farm','day']].copy();d['weight']=v;d['contribution']=v*ty;d=d.groupby(['farm','day'],as_index=False)[['weight','contribution']].sum();d=d[d.weight>0];d['weighted_ec']=d.contribution/d.weight;d['arm']=ARM;d['level']=level;supports.append(d)
    return profiles,pd.concat(supports,ignore_index=True)
def main():
    L.mkdir(parents=True,exist_ok=True);raw,jobs,prep=prepare()
    if '--prepare' in sys.argv:print('PREPARED_TWO_H0_FIT0',flush=True);return
    B.write(L/'worker.lock',dict(pid=os.getpid(),source_sha=B.sha(__file__)));frames=[];receipts=[];profiles=[];supports=[]
    base=pd.read_csv(B.L/'baseline_rows.csv',float_precision='round_trip',dtype={'seed':str})
    for k,(t,q,pfn) in jobs.items():
        rs={s:load_base(k,s,t,q) for s in B.SEEDS};ref=set(t[['farm','day']].itertuples(index=False,name=None));ec=t.groupby(['farm','day']).sub_ec.mean();structure=B.SG.prepare(raw,ref);cal=B.SG.ref_calendar(structure,ref);ets=[];r3=[]
        for s in B.SEEDS:
            assert B.sha(__file__)==prep['source_sha'] and B.sha(H/'core_v1.py')==prep['core_sha'];start=time.monotonic();dest=L/f'{k}_{s}.npz';assert not dest.exists()
            model=B.M.et(s)
            with threadpool_limits(limits=2):
                model.fit(t[COLS],t.sub_ec.to_numpy(float));model.steps[-1][1].n_jobs=1;et=np.asarray(model.predict(q[COLS]),float);assert np.array_equal(et[:8],model.predict(q.iloc[:8][COLS]))
            assert np.isfinite(et).all()
            if k==1 and s==7:
                p,d=save_trace(model,t,q);profiles.extend(p);supports.append(d)
            del model;gc.collect();np.savez_compressed(dest,row_id=q.row_id.to_numpy(str),train_row_id=t.row_id.to_numpy(str),et=et)
            meta=dict(k=k,seed=s,sha=B.sha(dest),prep_sha=B.sha(H/'preparation_v1.json'),source_sha=prep['source_sha'],train_hash=prep['records'][k]['train_hash'],query_hash=prep['records'][k]['query_hash'],seconds=time.monotonic()-start);B.write(dest.with_suffix('.json'),meta);receipts.append(meta);ets.append(et)
            mix=.6*et+.3*rs[s]['lgb']+.1*rs[s]['mlp'];r3.append(mix);f=B.post(k,ARM,s,t,q,.8*mix+.2*pfn,structure,cal,ec,ref);f['raw_et']=et;frames.append(f)
            b=base[(base.k==k)&base.seed.eq(str(s))].set_index('row_id').loc[q.row_id].reset_index();b['raw_et']=rs[s]['et'];frames.append(b)
            print('ET_TWO_H0_COMPLETE',k,s,round(meta['seconds'],1),flush=True)
        f=B.post(k,ARM,'ensemble',t,q,.8*np.mean(r3,axis=0)+.2*pfn,structure,cal,ec,ref);f['raw_et']=np.mean(ets,axis=0);frames.append(f)
        b=base[(base.k==k)&base.seed.eq('ensemble')].set_index('row_id').loc[q.row_id].reset_index();b['raw_et']=np.mean([rs[s]['et'] for s in B.SEEDS],axis=0);frames.append(b);print('TWO_H0_FOLD_COMPLETE',k,flush=True)
    rows=L/'rows.csv';assert not rows.exists();pd.concat(frames,ignore_index=True).to_csv(rows,index=False)
    pd.DataFrame(profiles).to_csv(H/'support_profiles_v1.csv',index=False);pd.concat(supports,ignore_index=True).to_csv(H/'support_days_v1.csv',index=False)
    B.write(H/'receipt_v1.json',dict(status='COMPLETE_TWO_H0_PUBLIC_DIAG10',new_ET_fits=30,other_new_fit=0,rows=69120,rows_sha=B.sha(rows),models=receipts,source_sha=prep['source_sha'],preparation_sha=B.sha(H/'preparation_v1.json'),trace_sha=B.sha(L/'trace.npz'),adoption=False))
    assert json.loads((L/'worker.lock').read_text())['pid']==os.getpid();(L/'worker.lock').unlink();print('TWO_H0_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()
