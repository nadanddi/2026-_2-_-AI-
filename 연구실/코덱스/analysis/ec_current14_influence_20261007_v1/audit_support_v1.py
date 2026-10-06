from pathlib import Path
import sys,json,math,gc
from collections import defaultdict
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import runner_v4 as R
import numpy as np,pandas as pd
def route(X,left,right,feature,threshold,y=None):
    n=len(left);nodes=np.zeros(len(X),np.int32);alive=np.arange(len(X));counts=np.zeros(n,np.int64);sums=np.zeros(n,float)
    for depth in range(n+1):
        if not len(alive):break
        here=nodes[alive];counts+=np.bincount(here,minlength=n)
        if y is not None:sums+=np.bincount(here,weights=y[alive],minlength=n)
        isleaf=left[here]==-1;assert np.all(right[here[isleaf]]==-1);alive=alive[~isleaf]
        if len(alive):
            here=nodes[alive];cols=feature[here];assert np.all((cols>=0)&(cols<X.shape[1]));nxt=np.where(X[alive,cols]<=threshold[here],left[here],right[here]);assert np.all((nxt>=0)&(nxt<n));nodes[alive]=nxt
    else:raise AssertionError('Malformed tree')
    return nodes,counts,sums
def main():
    raw,jobs,_=R.prepare();t,q,_=jobs[1];q=q[(q.farm=='F47')&(q.day==161)].sort_values('hour').reset_index(drop=True)
    profiles=pd.read_csv(H/'influence_support_profiles_v1.csv',float_precision='round_trip');days=pd.read_csv(H/'influence_support_days_v1.csv',float_precision='round_trip');out=[]
    for arm,deleted in [('BASE',[]),('D1',[139]),('D2',[139,231])]:
        path=R.L/'trace'/(arm+'.npz');meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert meta['sha']==R.sha(path)
        with np.load(path,allow_pickle=False) as c:a={n:c[n].copy() for n in c.files}
        tt=t[~(t.farm.eq('F47')&t.day.isin(deleted))].reset_index(drop=True)
        assert a['train_row_id'].tolist()==tt.row_id.tolist() and a['query_row_id'].tolist()==q.row_id.tolist()
        assert np.array_equal(a['train_y'],tt.sub_ec) and np.array_equal(a['query_y'],q.sub_ec)
        xt=tt[R.M.FULL_R3].to_numpy(float);qt=q[R.M.FULL_R3].to_numpy(float);med=np.nanmedian(xt,axis=0);np.testing.assert_allclose(med,a['imputer_median'],atol=1e-12,rtol=0)
        X=np.where(np.isnan(xt),med,xt).astype(np.float32);Q=np.where(np.isnan(qt),med,qt).astype(np.float32)
        assert np.array_equal(X,a['train_X']) and np.array_equal(Q,a['query_X']);assert X.shape[1]==47
        off=a['offsets'];assert len(off)==601 and off[0]==0 and np.all(np.diff(off)>0) and off[-1]==meta['nodes'];weights=np.zeros((len(q),len(tt)));tp=np.empty((len(q),600));nodeerr=0
        for tree in range(600):
            lo,hi=map(int,off[tree:tree+2]);left,right,feature,threshold=[a[n][lo:hi] for n in ['left','right','feature','threshold']]
            tl,cnt,ysum=route(X,left,right,feature,threshold,a['train_y']);ql,_,_=route(Q,left,right,feature,threshold)
            assert np.array_equal(cnt,a['n_samples'][lo:hi]) and np.array_equal(cnt,a['weighted_n_samples'][lo:hi]);assert np.all(cnt>0)
            err=float(np.max(abs(ysum/cnt-a['value'][lo:hi])));nodeerr=max(nodeerr,err);assert err<1e-10
            assert np.array_equal(tl,a['train_leaf'][:,tree]) and np.array_equal(ql,a['query_leaf'][:,tree]);tp[:,tree]=a['value'][lo:hi][ql]
            cohorts=defaultdict(list)
            for i,leaf in enumerate(tl):cohorts[int(leaf)].append(i)
            for i,leaf in enumerate(ql):ix=cohorts[int(leaf)];assert len(ix);weights[i,ix]+=1/(600*len(ix))
        np.testing.assert_allclose(weights,a['weights'],atol=1e-12,rtol=0);assert weights.min()>=0;np.testing.assert_allclose(weights.sum(axis=1),1,atol=1e-12,rtol=0)
        pred=np.array([math.fsum(v)/600 for v in tp]);np.testing.assert_allclose(pred,a['raw_et'],atol=1e-10,rtol=0);np.testing.assert_allclose(weights@a['train_y'],pred,atol=1e-10,rtol=0)
        cp=R.L/('base/1_7.npz' if arm=='BASE' else f'ablation/{arm}_1_7.npz')
        with np.load(cp,allow_pickle=False) as c:ix=pd.Index(c['row_id']).get_indexer(q.row_id);assert min(ix)>=0;replay=float(np.max(abs(pred-c['et'][ix])));assert replay<1e-10
        ym=tt.groupby(['farm','day']).sub_ec.transform('mean').to_numpy(float);sw=np.array([.5*weights[i]+.5*weights[:i+1].mean(axis=0) for i in range(24)])
        for level,v in [('hour0',weights[0]),('raw_day',weights.mean(axis=0)),('smooth_day',sw.mean(axis=0))]:
            p=profiles[(profiles.arm==arm)&(profiles.level==level)];assert len(p)==1;p=p.iloc[0]
            expected=dict(prediction=float(v@a['train_y']),truth=float(a['query_y'][0] if level=='hour0' else a['query_y'].mean()),high_day_weight=float(v[ym>=1].sum()),high_row_weight=float(v[a['train_y']>=1].sum()))
            for n,val in expected.items():assert math.isclose(val,float(p[n]),abs_tol=1e-10,rel_tol=0)
            d=tt[['farm','day']].copy();d['weight']=v;d['contribution']=v*a['train_y'];d=d.groupby(['farm','day'],as_index=False)[['weight','contribution']].sum();d=d[d.weight>0].set_index(['farm','day']);r=days[(days.arm==arm)&(days.level==level)].set_index(['farm','day']);assert set(d.index)==set(r.index)
            for key,row in d.iterrows():
                for n in ['weight','contribution']:assert math.isclose(float(row[n]),float(r.loc[key,n]),abs_tol=1e-10,rel_tol=0)
                assert math.isclose(float(row.contribution/row.weight),float(r.loc[key,'weighted_ec']),abs_tol=1e-10,rel_tol=0)
        out.append(dict(arm=arm,trees=600,nodes=int(off[-1]),train_rows=len(tt),query_rows=24,node_mean_maxdiff=nodeerr,replay_maxdiff=replay,sha=R.sha(path)));print('SUPPORT_AUDIT_PASS',arm,int(off[-1]),flush=True);del a,X,Q,weights,tp;gc.collect()
    R.write(H/'support_audit_v1.json',dict(status='PASS_ALL_SAVED_TREE_PATHS_AND_SUPPORT',new_fit=0,models=out,total_nodes=sum(m['nodes'] for m in out),source_sha=R.sha(__file__)))
if __name__=='__main__':main()
