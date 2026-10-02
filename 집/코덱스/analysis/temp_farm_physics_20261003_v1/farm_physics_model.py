from pathlib import Path
import importlib.util
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('temp_base_reference',H.parent/'temp_hierarchical_20261003_v1/run_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def transform(d):
    x=d[m.PHYSICS_COLUMNS].copy();f=d.farm_id.to_numpy()-.5
    for c in m.PHYSICS_COLUMNS:
        if c!='farm_id':x[c+'_farm_slope']=d[c].to_numpy()*f
    return x
def fit_fold(tr,va,w,seed,guard=False):
    assert not guard
    lin=m.make_pipeline(m.SimpleImputer(strategy='median',keep_empty_features=True),m.StandardScaler(),m.Ridge(alpha=100.))
    lin.fit(transform(tr),tr.sub_temp.to_numpy(),ridge__sample_weight=w);pt,pv=lin.predict(transform(tr)),lin.predict(transform(va))
    tree=m.LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=seed)
    tree.fit(tr[m.FEATURE_COLUMNS],tr.sub_temp.to_numpy()-pt,sample_weight=w);p=pv+tree.predict(va[m.FEATURE_COLUMNS]);q=pt+tree.predict(tr[m.FEATURE_COLUMNS])
    stats={'F1':dict(train_rmse=m.rmse(q,tr.sub_temp.to_numpy()),validation_rmse=m.rmse(p,va.sub_temp.to_numpy()),training_days=len(tr[['farm','day']].drop_duplicates()),training_rows=len(tr))}
    return {'F1':p,'physics':pv},stats
