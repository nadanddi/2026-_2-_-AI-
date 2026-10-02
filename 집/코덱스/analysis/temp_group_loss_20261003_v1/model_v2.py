from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference_experiment',HERE.parent/'temp_hierarchical_20261003_v1/run_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
np=m.np

def loss_parts(e,w,groups,kind):
    totals=np.bincount(groups,weights=w);means=np.bincount(groups,weights=w*e)/totals
    if kind in ['L1','S1','ZERO']:
        lam={'L1':1.,'S1':-.5,'ZERO':0.}[kind]
        loss=.5*np.sum(w*e*e)+.5*lam*np.sum(totals*means*means)
        grad=w*(e+lam*means[groups]);hess=w*(1+lam*w/totals[groups])
    else:
        edge=groups[1:]==groups[:-1];v=np.minimum(w[1:],w[:-1])*edge;d=e[1:]-e[:-1]
        loss=.5*np.sum(w*e*e)+.5*np.sum(v*d*d);grad=w*e;hess=w.copy()
        grad[1:]+=v*d;grad[:-1]-=v*d;hess[1:]+=v;hess[:-1]+=v
    return float(loss),grad,hess

def fit_fold(tr,va,w,seed,guard=False):
    assert not guard
    lin=m.make_pipeline(m.SimpleImputer(strategy='median',keep_empty_features=True),m.StandardScaler(),m.Ridge(alpha=100.))
    lin.fit(tr[m.PHYSICS_COLUMNS],tr.sub_temp.to_numpy(),ridge__sample_weight=w)
    pt,pv=lin.predict(tr[m.PHYSICS_COLUMNS]),lin.predict(va[m.PHYSICS_COLUMNS]);res=tr.sub_temp.to_numpy()-pt
    intercept=float(np.average(res,weights=w));target=res-intercept
    groups=m.pd.factorize(m.pd.MultiIndex.from_frame(tr[['farm','day']]),sort=False)[0]
    # Same-day consecutive hourly rows only; no temporal edge across days/gaps.
    assert np.all(tr.hour.to_numpy()[1:][groups[1:]==groups[:-1]]-tr.hour.to_numpy()[:-1][groups[1:]==groups[:-1]]==1)
    result={'physics':pv};stats={}
    for tag in ['L1','S1','D1']:
        def objective(y,p):
            _,grad,hess=loss_parts(p-y,w,groups,tag);return grad,hess
        tree=m.LGBMRegressor(objective=objective,n_estimators=220,learning_rate=.035,num_leaves=12,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=seed,boost_from_average=False)
        tree.fit(tr[m.FEATURE_COLUMNS],target)
        train=pt+intercept+tree.predict(tr[m.FEATURE_COLUMNS]);result[tag]=pv+intercept+tree.predict(va[m.FEATURE_COLUMNS])
        stats[tag]=dict(train_rmse=m.rmse(train,tr.sub_temp.to_numpy()),validation_rmse=m.rmse(result[tag],va.sub_temp.to_numpy()),residual_intercept=intercept,training_days=len(np.unique(groups)),training_rows=len(tr))
    return result,stats
