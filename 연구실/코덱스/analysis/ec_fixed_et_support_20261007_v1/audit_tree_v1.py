from pathlib import Path
import sys,json,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as R
sys.path.insert(0,str(R.B.H));from audit_support_v1 import route
import numpy as np
def main():
    records=[]
    for label in ['original_4_7','original_8_7','common_7']:
        path=R.L/f'{label}.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert meta['sha']==R.B.sha(path)
        with np.load(path) as z:d={k:z[k].copy() for k in z.files}
        inputs=[];leaves=[]
        pp=sorted(R.L.glob(f'coalition_{label}_*.npz'));assert len(pp)=={'original_4_7':1,'original_8_7':2,'common_7':5}[label]
        for p in pp:
            assert json.loads(p.with_suffix('.json').read_text())['sha']==R.B.sha(p)
            with np.load(p) as z:inputs.append(z['X'].copy());leaves.append(z['query_leaf'].copy())
        Q=np.concatenate(inputs);ql=np.concatenate(leaves);maxerr=0
        for j,(a,b) in enumerate(zip(d['offsets'][:-1],d['offsets'][1:])):
            left,right,feature,threshold=[d[n][a:b] for n in ['left','right','feature','threshold']]
            tl,cnt,ysum=route(d['train_X'],left,right,feature,threshold,d['train_y']);qq,_,_=route(Q,left,right,feature,threshold)
            assert np.array_equal(tl,d['train_leaf'][:,j]) and np.array_equal(qq,ql[:,j]);assert np.array_equal(cnt,d['n_samples'][a:b]) and np.array_equal(cnt,d['weighted_n_samples'][a:b]);assert cnt.min()>0
            err=float(np.max(abs(ysum/cnt-d['value'][a:b])));assert err<1e-10;maxerr=max(maxerr,err)
        records.append(dict(model=label,trees=600,nodes=int(d['offsets'][-1]),all_query_rows=len(Q),node_mean_maxdiff=maxerr,forest_sha=R.B.sha(path)));print('FULL_TREE_AUDIT_PASS',label,records[-1],flush=True);del d,Q,ql,inputs,leaves;gc.collect()
    R.write(H/'audit_tree_v1.json',dict(status='PASS',fit=0,new_seed7_fullforests=records,total_nodes=sum(r['nodes'] for r in records),source_sha=R.B.sha(__file__)))
if __name__=='__main__':main()
