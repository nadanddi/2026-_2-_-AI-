from pathlib import Path
import sys,json,gc,math
from collections import defaultdict
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as R
sys.path.insert(0,str(R.B.H));from audit_support_v1 import route
import numpy as np,pandas as pd
def main():
    complete=json.loads((H/'completion_v1.json').read_text(encoding='utf-8'));assert complete['new_ET_fits']==5
    tree=json.loads((H/'audit_tree_v1.json').read_text(encoding='utf-8'));assert tree['status']=='PASS';records=tree['new_seed7_fullforests']
    for r in records:assert r['forest_sha']==R.B.sha(R.L/(r['model']+'.npz'))
    endpoint_checks=[];path_rows=0
    for fp in sorted(R.L.glob('original_*.npz'))+sorted(R.L.glob('common_*.npz')):
        label=fp.stem
        with np.load(fp) as z:d={k:z[k].copy() for k in z.files}
        for p in sorted(R.L.glob(f'coalition_{label}_*.npz')):
            with np.load(p) as z:v={k:z[k].copy() for k in z.files}
            assert np.array_equal(v['train_row_id'],d['train_row_id']);assert (v['weights_smooth']>=0).all();assert np.max(abs(v['weights_smooth'].sum(axis=1)-1))<1e-12
            for key,w in [('raw',v['weights_raw']),('smooth',v['weights_smooth'])]:assert np.max(abs(w@d['train_y']-v[key]))<1e-10
            for end,index in [('control',0),('failure',-1)]:
                leaf=v['query_leaf'].reshape(-1,24,600)[index];point=np.zeros((24,len(d['train_y'])))
                for j in range(600):
                    cohorts=defaultdict(list)
                    for row,lf in enumerate(d['train_leaf'][:,j]):cohorts[int(lf)].append(row)
                    for hour,lf in enumerate(leaf[:,j]):ix=cohorts[int(lf)];point[hour,ix]+=1/(600*len(ix))
                smooth=.5*point+.5*np.cumsum(point,axis=0)/np.arange(1,25)[:,None]
                assert np.max(abs(smooth.mean(axis=0)-v['weights_smooth'][index]))<1e-12
                assert np.max(abs(point.mean(axis=0)-v['weights_raw'][index]))<1e-12
                endpoint_checks.append(dict(model=label,pair=p.stem,endpoint=end,maxdiff=float(np.max(abs(smooth.mean(axis=0)-v['weights_smooth'][index])))))
        for p in sorted(H.glob(f'paths_{label}_*.json')):
            rr=json.loads(p.read_text(encoding='utf-8'));assert rr['tree_denominator']==600;assert len(rr['records'])==2400
            tag=rr['records'][0]['pair']
            with np.load(R.L/f'coalition_{label}_{tag}.npz') as z:X=z['X'].reshape(-1,24,47);A=X[0];T=X[-1]
            for r in rr['records']:
                a=d['offsets'][r['tree']];hour=r['hour'];node=0;div=False
                while d['left'][a+node]!=-1:
                    f=int(d['feature'][a+node]);th=d['threshold'][a+node];cl=A[hour,f]<=th;tl=T[hour,f]<=th
                    if cl!=tl:div=True;break
                    node=int(d['left'][a+node] if cl else d['right'][a+node])
                assert div==r['diverged']
                if div:
                    assert node==r['node'] and R.COLS[f]==r['column'] and th==r['threshold'];assert len(r['prefix'])>=0
                    for child,key in [(r['control_child'],'control'),(r['target_child'],'target')]:assert d['n_samples'][a+child]==r[key+'_count'] and d['value'][a+child]==r[key+'_node_mean']
                path_rows+=1
        del d;gc.collect();print('ENDPOINT_PATH_AUDIT_PASS',label,flush=True)
    R.write(H/'audit_endpoint_v2.json',dict(status='PASS',fit=0,new_seed7_fullforests=records,total_nodes=sum(r['nodes'] for r in records),endpoint_support_checks=endpoint_checks,path_rows=path_rows,other_seeds_full_routing=False,existing_original1_full_routing=False,source_sha=R.B.sha(__file__)))
if __name__=='__main__':main()
