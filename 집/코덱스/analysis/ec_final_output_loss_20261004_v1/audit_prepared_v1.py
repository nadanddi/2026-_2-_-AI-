"""Root independent pre-fit audit of heldout IDs, features and preprocessing.
No EC training/model prediction/scoring. Public cached baselines are formula checked.
"""
from pathlib import Path
import sys,json,ast,hashlib,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(H));import verify_full_v2 as V
M,np,pd,S=V.M,V.np,V.pd,V.S
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def main():
    p=V.readj(H/'preparation_v3.json');assert p['source_sha256']==V.sha(H/'run_v3.py')
    assert p['source_sha256']=='be6bab3303e37decad5bd078e56edf569cda5bf0baaea470df971bc53043b172'
    assert V.sha(H/'preparation_v3.json')=='df4ff2c3b7617676c61e4cee4b8e9690f9312a7cca448c0caec189cb5051b860'
    assert all(p[k]==0 for k in ('model_fit_count','model_predict_count','real_score_count'))
    assert p['config']['epochs']==400 and p['config']['alpha']==.025/24
    lab,core,wv,folds,outer=S.loadec();cols=[c for c in core.BASE if c!='day']+['season']
    assert len(cols)==14 and not set(cols)&{'in_rad','act_side','act_valve','act_cool','act_pump'}
    assert [(v,k) for v,k,_,_ in folds]==V.FOLDS
    inner=V.extract(V.NPATH,{'arrayfile','full_inner'},dict(np=np,S=S,INNER=V.INNER))['full_inner']
    records=[];cachecount=0
    for j,(v,k,tm,vm) in enumerate(folds):
        sig=p['manifest'][j];a,b,z,bag=inner(v,k,lab[tm].reset_index(drop=True),lab.set_index('row_id'))
        a,b=S.seasonal(a,b,wv);tr,q=S.seasonal(lab[tm],lab[vm],wv)
        a,b,tr,q=[f.reset_index(drop=True) for f in (a,b,tr,q)]
        for name,f in [('inner_a',a),('inner_b',b),('outer_train',tr),('outer_query',q)]:
            assert sig[name+'_ids']==V.ids(f.row_id) and sig[name+'_features']==V.ah(f[cols].to_numpy(float)) and sig[name+'_targets']==V.ah(f.sub_ec.to_numpy(float))
        for rel,h in sig['cache_hashes'].items():assert V.sha(ROOT/rel)==h;cachecount+=1
        x=b[cols].to_numpy(float);imp=SimpleImputer(strategy='median',keep_empty_features=True);filled=imp.fit_transform(x)
        sc=StandardScaler();xb=sc.fit_transform(filled)
        stats=dict(imputer_statistics=imp.statistics_,scaler_mean=sc.mean_,scaler_scale=sc.scale_,scaler_var=sc.var_,scaler_n_samples_seen=np.asarray(sc.n_samples_seen_))
        assert {key:V.ah(value) for key,value in stats.items()}==sig['preprocessing']
        assert V.ah(xb)==sig['inner_b_transformed_features']
        xx=q[cols].to_numpy(float).copy();mask=np.isnan(xx);xx[mask]=np.broadcast_to(imp.statistics_,xx.shape)[mask]
        assert V.ah((xx-sc.mean_)/sc.scale_)==sig['outer_query_transformed_features']
        # Zero-weights fixture verifies independent median/moments before any real model.
        fixture=dict(stats,feature_names=np.asarray(cols),**{f'model__{layer}.{part}':np.zeros(shape) for layer,shapeW,shapeB in [(0,(64,14),(64,)),(2,(32,64),(32,)),(4,(1,32),(1,))] for part,shape in [('weight',shapeW),('bias',shapeB)]})
        assert np.array_equal(M.replay_checkpoint(fixture,x,q[cols].to_numpy(float),cols),np.zeros(len(q)))
        old=V.arrays(V.BASE/f'{v}_{k}_baseline.npz');assert np.array_equal(old['row_id'],q.row_id)
        for seed in V.SEEDS:
            raw=.8*old[f'r3_{seed}']+.2*old['old_pfn_raw']
            M.near(M.scalar_final(q,raw,float(old['lo']),float(old['hi'])),old[f'baseline_{seed}'])
        records.append(dict(validator=v,fold=k,inner_a=len(a),inner_b=len(b),outer_query=len(q)))
    result=dict(status='PASS_INDEPENDENT_PREPARED_IDS_FEATURES_MOMENTS',records=records,cache_hash_checks=cachecount,checks=M.CHECKS,source_sha256=V.sha(Path(__file__)),runner_sha256=p['source_sha256'],preparation_sha256=V.sha(H/'preparation_v3.json'),real_model_fit=0,real_model_predict=0,real_score=0,preprocessor_fit=22,zero_fixture_replays=22,limitations=['No optimization performed; zero fixture is not a trained EC model','Existing inner/full guard source reused; preprocessor moments independently recomputed'])
    V.save(H/'independent_preparation_v1.json',result);print('PASS_INDEPENDENT_PREPARATION',cachecount,M.CHECKS)
if __name__=='__main__':main()
