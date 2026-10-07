"""Same feature grammar, vectorized rolling arrays; v1 remains independent reference."""
from pathlib import Path
import numpy as np
path=Path(__file__).with_name('causal_features_v1.py')
code=path.read_text(encoding='utf-8')
old='''            means=[];stds=[];sums=[];counts=[]
            for hour in h:
                b=s.reindex(np.arange(max(0,hour-w+1),hour+1)).to_numpy()
                n=np.isfinite(b).sum();counts.append(n)
                means.append(np.nanmean(b) if n else np.nan)
                sums.append(np.nansum(b) if n else np.nan)
                stds.append(np.nanstd(b) if n else np.nan)
'''
new='''            dense=s.reindex(np.arange(24)).to_numpy()
            padded=np.pad(dense,(w-1,0),constant_values=np.nan)
            matrix=np.lib.stride_tricks.sliding_window_view(padded,w)[h]
            observed=np.isfinite(matrix);counts=observed.sum(axis=1)
            sums=np.where(observed,matrix,0).sum(axis=1)
            means=np.divide(sums,counts,out=np.full(len(h),np.nan),where=counts>0)
            squared=np.where(observed,(matrix-means[:,None])**2,0).sum(axis=1)
            stds=np.sqrt(np.divide(squared,counts,out=np.full(len(h),np.nan),where=counts>0))
            sums=np.where(counts>0,sums,np.nan)
'''
assert old in code
namespace={'__file__':str(path),'__name__':'fast_causal_features_v2'}
exec(compile(code.replace(old,new),str(Path(__file__).resolve()),'exec'),namespace)
build=namespace['build'];RAW=namespace['RAW'];annotate=namespace['annotate']

if __name__=='__main__':
    import json,time
    import pandas as pd
    from causal_features_v1 import build as reference
    here=Path(__file__).resolve().parent
    import env
    X=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+RAW)
    X=X[X.row_id.str.startswith(('F13_','F47_'))].copy()
    t=time.perf_counter();F,meta=build(X)
    compare_count=0
    def compare(a,b,tolerance=0):
        global compare_count
        np.testing.assert_allclose(a.to_numpy(),b.to_numpy(),rtol=0,atol=tolerance,equal_nan=True)
        assert a.index.equals(b.index) and a.columns.equals(b.columns)
        compare_count+=1
    for prefix in ['F13_008_','F47_006_']:
        G=X[X.row_id.str.startswith(prefix)]
        A,_=reference(G);B,_=build(G);compare(A,B,1e-10)
        for hour in [0,1,2,3,4,5,6,12,23]:
            hours=G.row_id.str[-2:].astype(int);ids=G.loc[hours<=hour,'row_id']
            P,_=build(G[hours<=hour]);compare(F.loc[P.index],P)
            Q=G.copy();Q.loc[hours>hour,RAW]=99999
            altered,_=build(Q);compare(F.loc[ids],altered.loc[ids])
    shuffled,_=build(X.sample(frac=1,random_state=47));compare(F,shuffled)
    other=X.copy();other.loc[other.row_id.str.startswith('F47_'),RAW]=77777
    O,_=build(other);ids=F.index[F.index.str.startswith('F13_')];compare(F.loc[ids],O.loc[ids])
    assert not np.isinf(F.to_numpy()).any()
    result={'status':'PASS','rows':len(F),'columns':len(F.columns),'comparisons':compare_count,
        'reference_method':'v1 pandas逐시각 reindex/mean/std 대조; 2일만, 전체 v1 감사 별도 진행',
        'future_prefix_atol':0,'reference_atol':1e-10,'CPU_only':True,'elapsed_seconds':time.perf_counter()-t,
        'performance_tested':False,'whole_model_causality_tested':False}
    p=here/'fast_feature_audit_v2.json';assert not p.exists()
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    p=here/'feature_definitions_v2.json';assert not p.exists()
    p.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
