from pathlib import Path
import importlib.util
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('temp_pipeline_reference',H.parent/'temp_hierarchical_20261003_v1/run_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
np=m.np
def physics(tr,va,w,target=None):
    lin=m.make_pipeline(m.SimpleImputer(strategy='median',keep_empty_features=True),m.StandardScaler(),m.Ridge(alpha=100.))
    lin.fit(tr[m.PHYSICS_COLUMNS],tr.sub_temp.to_numpy() if target is None else target,ridge__sample_weight=w)
    return lin.predict(tr[m.PHYSICS_COLUMNS]),lin.predict(va[m.PHYSICS_COLUMNS])
def tree(tr,target,w,seed,cols):
    t=m.LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=seed)
    t.fit(tr[cols],target,sample_weight=w);return t
def relative(d):
    x=d.copy();cols=[]
    for c in ['in_temp_h0','in_temp_mean','in_temp_reset1','in_temp_reset3','in_temp_reset8']:
        k=c+'_minus_current';x[k]=x[c]-x.in_temp;cols.append(k)
    k='in_temp_reset3_minus_reset8';x[k]=x.in_temp_reset3-x.in_temp_reset8;cols.append(k)
    return x,cols
def inner_folds(tr):
    days={f:sorted(tr.loc[tr.farm==f,'day'].unique().tolist()) for f in m.common.TARGET_FARMS}
    return [{f:{int(d) for i,d in enumerate(days[f]) if (i//5)%3==k} for f in days} for k in range(3)]
def fit_fold(tr,va,w,seed,guard=False):
    assert not guard
    y=tr.sub_temp.to_numpy();py=va.sub_temp.to_numpy();pt,pv=physics(tr,va,w);base=tree(tr,y-pt,w,seed,m.FEATURE_COLUMNS);rt,rv=base.predict(tr[m.FEATURE_COLUMNS]),base.predict(va[m.FEATURE_COLUMNS])
    ip=np.full(len(tr),np.nan);ir=np.full(len(tr),np.nan);audit=[]
    for k,fd in enumerate(inner_folds(tr)):
        tm,vm=m.common.split_mask(tr,fd);a,b=tr[tm].copy(),tr[vm].copy();bp,bv=physics(a,b,w[tm]);t=tree(a,a.sub_temp.to_numpy()-bp,w[tm],seed,m.FEATURE_COLUMNS);ip[vm]=bv;ir[vm]=t.predict(b[m.FEATURE_COLUMNS])
        train_days=sorted(set(zip(a.farm,a.day.astype(int))));query_days=sorted(set(zip(b.farm,b.day.astype(int))));assert all(f!=g or abs(d-e)>1 for f,d in train_days for g,e in query_days)
        audit.append(dict(fold=k,training_days=train_days,query_days=query_days,training_rows=len(a),query_rows=len(b)))
    assert np.isfinite(ip).all() and np.isfinite(ir).all()
    a=tree(tr,y-ip,w,seed,m.FEATURE_COLUMNS);pa=pv+a.predict(va[m.FEATURE_COLUMNS]);qa=pt+a.predict(tr[m.FEATURE_COLUMNS])
    ai=m.SimpleImputer(strategy='median',keep_empty_features=True).fit(tr[['in_temp_reset3']]);at=ai.transform(tr[['in_temp_reset3']]).ravel();av=ai.transform(va[['in_temp_reset3']]).ravel()
    pbt,pbv=physics(tr,va,w,y-at);pbt+=at;pbv+=av;b=tree(tr,y-pbt,w,seed,m.FEATURE_COLUMNS);pb=pbv+b.predict(va[m.FEATURE_COLUMNS]);qb=pbt+b.predict(tr[m.FEATURE_COLUMNS])
    xt,cols=relative(tr);xv,_=relative(va);fc=m.FEATURE_COLUMNS+cols;c=tree(xt,y-pt,w,seed,fc);pc=pv+c.predict(xv[fc]);qc=pt+c.predict(xt[fc])
    # Train-only nested OOF calibration, regularized towards coefficient 1.
    numerator=float(np.sum(w*ir*(y-ip))+100.);denominator=float(np.sum(w*ir*ir)+100.);gamma=float(np.clip(numerator/denominator,0.,1.25));pd=pv+gamma*rv;qd=pt+gamma*rt
    result={'A1':pa,'B1':pb,'C1':pc,'I1':pd,'physics':pv,'cal_y':y,'cal_w':w,'cal_physics':ip,'cal_residual':ir,'cal_row_id':tr.row_id.to_numpy(dtype=str),'original_CODEX':pv+rv}
    stats={tag:dict(train_rmse=m.rmse(q,y),validation_rmse=m.rmse(p,py),training_days=len(tr[['farm','day']].drop_duplicates()),training_rows=len(tr)) for tag,p,q in [('A1',pa,qa),('B1',pb,qb),('C1',pc,qc),('I1',pd,qd)]}
    stats['A1']['inner_split_audit']=audit;stats['I1'].update(gamma=gamma,numerator=numerator,denominator=denominator,inner_original_rmse=m.rmse(ip+ir,y),inner_scaled_rmse=m.rmse(ip+gamma*ir,y))
    return result,stats
