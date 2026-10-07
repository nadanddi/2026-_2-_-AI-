from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'연구실/코덱스/local'/H.name
prep=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'));cols=pd.read_csv(H/'profiles_v1.csv',nrows=1).columns.tolist()[5:]
assert len(cols)==47
def main():
    rows=0;divergences=0
    for label in ['original_1_7','original_4_7','original_8_7','common_7']:
        with np.load(L/f'{label}.npz') as z:d={k:z[k].copy() for k in ['offsets','left','right','feature','threshold','value','n_samples']}
        for fp in H.glob(f'paths_{label}_*.json'):
            records=json.loads(fp.read_text(encoding='utf-8'))['records'];tag=records[0]['pair']
            with np.load(L/f'coalition_{label}_{tag}.npz') as z:X=z['X'].reshape(-1,24,47);A=X[0];T=X[-1]
            for r in records:
                rows+=1;node=0;prefix=[];a=d['offsets'][r['tree']];hour=r['hour'];div=False
                while d['left'][a+node]!=-1:
                    f=int(d['feature'][a+node]);th=float(d['threshold'][a+node]);cl=bool(A[hour,f]<=th);tl=bool(T[hour,f]<=th)
                    if cl!=tl:div=True;break
                    prefix.append([node,cols[f],th,cl]);node=int(d['left'][a+node] if cl else d['right'][a+node])
                assert div==r['diverged']
                if div:
                    divergences+=1;assert prefix==r['prefix'];assert r['column']==cols[f] and r['control_value']==A[hour,f] and r['target_value']==T[hour,f] and r['control_left']==cl
                    assert r['control_child']==int(d['left'][a+node] if cl else d['right'][a+node]);assert r['target_child']==int(d['left'][a+node] if tl else d['right'][a+node])
        del d
    common=[]
    for seed in [7,101,2024]:
        with np.load(L/f'common_{seed}.npz') as z:common.append({k:z[k].copy() for k in ['train_row_id','train_X','train_y','imputer_median']})
    for d in common[1:]:
        for k in d:assert np.array_equal(d[k],common[0][k])
    e=pd.read_csv(H/'endpoints_v1.csv');count=int(e[e.endpoint=='control'].coalitions.sum());assert count==18432
    with (H/'audit_prefix_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS',fit=0,path_rows=rows,divergent_paths=divergences,common_3seed_train_imputer_equal=True,coalitions=count),f,indent=2)
    print('PREFIX_COMMON_AUDIT_PASS',rows,divergences,count)
if __name__=='__main__':main()
