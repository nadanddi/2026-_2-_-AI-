from pathlib import Path
import sys,json,math
from collections import defaultdict
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as R
sys.path.insert(0,str(R.OLD));from audit_support_v1 import route
np,pd,B=R.np,R.pd,R.B
def main():
    _,jobs,_=R.prepare();t,q,_=jobs[1];q=q[(q.farm=='F47')&(q.day==161)].sort_values('hour').reset_index(drop=True);path=R.L/'trace.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert meta['sha']==B.sha(path) and meta['source_sha']==B.sha(H/'run_v1.py') and meta['prep_sha']==B.sha(H/'preparation_v1.json') and meta['seed']==7 and meta['k']==1 and meta['columns']==R.COLS
    with np.load(path,allow_pickle=False) as c:a={n:c[n].copy() for n in c.files}
    assert a['columns'].tolist()==R.COLS and a['train_row_id'].tolist()==t.row_id.tolist() and a['query_row_id'].tolist()==q.row_id.tolist();assert np.array_equal(a['train_y'],t.sub_ec) and np.array_equal(a['query_y'],q.sub_ec)
    xt=t[R.COLS].to_numpy(float);qt=q[R.COLS].to_numpy(float);med=np.nanmedian(xt,axis=0);assert np.isfinite(med).all();np.testing.assert_allclose(med,a['imputer_median'],atol=1e-12,rtol=0)
    with np.load(B.L/'trace'/'BASE.npz',allow_pickle=False) as old:
        assert old['train_row_id'].tolist()==a['train_row_id'].tolist();np.testing.assert_array_equal(old['imputer_median'][[B.M.FULL_R3.index(c) for c in R.COLS]],med)
    X=np.where(np.isnan(xt),med,xt).astype(np.float32);Q=np.where(np.isnan(qt),med,qt).astype(np.float32);assert X.shape[1]==45 and np.array_equal(X,a['train_X']) and np.array_equal(Q,a['query_X']);off=a['offsets'];assert len(off)==601 and off[0]==0 and np.all(np.diff(off)>0) and off[-1]==meta['nodes'];w=np.zeros((24,len(t)));tp=np.empty((24,600));nodeerr=0
    for tree in range(600):
        lo,hi=map(int,off[tree:tree+2]);left,right,feature,threshold=[a[n][lo:hi] for n in ['left','right','feature','threshold']];tl,cnt,ysum=route(X,left,right,feature,threshold,a['train_y']);ql,_,_=route(Q,left,right,feature,threshold)
        assert np.all(cnt>0) and np.array_equal(cnt,a['n_samples'][lo:hi]) and np.array_equal(cnt,a['weighted_n_samples'][lo:hi]);err=float(np.max(abs(ysum/cnt-a['value'][lo:hi])));nodeerr=max(nodeerr,err);assert err<1e-10
        assert np.array_equal(tl,a['train_leaf'][:,tree]) and np.array_equal(ql,a['query_leaf'][:,tree]);tp[:,tree]=a['value'][lo:hi][ql];cohorts=defaultdict(list)
        for i,leaf in enumerate(tl):cohorts[int(leaf)].append(i)
        for i,leaf in enumerate(ql):ix=cohorts[int(leaf)];w[i,ix]+=1/(600*len(ix))
    np.testing.assert_allclose(w,a['weights'],atol=1e-12,rtol=0);np.testing.assert_allclose(w.sum(axis=1),1,atol=1e-12,rtol=0);pred=np.array([math.fsum(v)/600 for v in tp]);np.testing.assert_allclose(pred,a['raw_et'],atol=1e-10,rtol=0);np.testing.assert_allclose(w@a['train_y'],pred,atol=1e-10,rtol=0)
    with np.load(R.L/'1_7.npz',allow_pickle=False) as c:ix=pd.Index(c['row_id']).get_indexer(q.row_id);replay=float(np.max(abs(c['et'][ix]-pred)));assert min(ix)>=0 and replay<1e-10
    profiles=pd.read_csv(H/'support_profiles_v1.csv',float_precision='round_trip');support=pd.read_csv(H/'support_days_v1.csv',float_precision='round_trip');ym=t.groupby(['farm','day']).sub_ec.transform('mean').to_numpy(float);sw=np.array([.5*w[i]+.5*w[:i+1].mean(axis=0) for i in range(24)])
    for level,v in [('hour0',w[0]),('raw_day',w.mean(axis=0)),('smooth_day',sw.mean(axis=0))]:
        p=profiles[profiles.level.eq(level)].iloc[0];vals=dict(prediction=float(v@a['train_y']),truth=float(a['query_y'][0] if level=='hour0' else a['query_y'].mean()),high_day_weight=float(v[ym>=1].sum()),high_row_weight=float(v[a['train_y']>=1].sum()))
        for key,val in vals.items():assert math.isclose(val,float(p[key]),abs_tol=1e-10,rel_tol=0)
        d=t[['farm','day']].copy();d['weight']=v;d['contribution']=v*a['train_y'];d=d.groupby(['farm','day'],as_index=False)[['weight','contribution']].sum();d=d[d.weight>0].set_index(['farm','day']);s=support[support.level.eq(level)].set_index(['farm','day']);assert set(d.index)==set(s.index)
        for key,r in d.iterrows():
            for name in ['weight','contribution']:assert math.isclose(float(r[name]),float(s.loc[key,name]),abs_tol=1e-10,rel_tol=0)
            assert math.isclose(float(r.contribution/r.weight),float(s.loc[key,'weighted_ec']),abs_tol=1e-10,rel_tol=0)
    B.write(H/'tree_audit_v1.json',dict(status='PASS_FULL_600_TREE_PATHS_SUPPORT_AND_RETAINED_MEDIAN',trees=600,nodes=int(off[-1]),node_mean_maxdiff=nodeerr,ET_replay_maxdiff=replay,new_fit=0,source_sha=B.sha(__file__),trace_sha=B.sha(path)));print('TREE_AUDIT_TWO_H0_PASS',int(off[-1]),flush=True)
if __name__=='__main__':main()
