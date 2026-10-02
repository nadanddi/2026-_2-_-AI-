import run_v2
R=run_v2.run
def main():
    R.common.load_raw=R.TM.masked_loader
    try:lab,_,_=R.TM.build_world()
    finally:R.common.load_raw=R.TM.ORIG;R.harness._CACHE.clear()
    tx,_,sx=R.TM.masked_loader();cf=R.build_features(tx,sx).set_index('row_id')
    for c in R.FEATURE_COLUMNS:
        if c not in lab:lab[c]=cf.loc[lab.row_id,c].to_numpy()
    w=R.TF.row_weights(lab,.2,w_noisy=.2);tm,vm=R.common.split_mask(lab,R.diag_folds(lab)[0]);tr,va=lab[tm],lab[vm];cache=dict(R.np.load(R.OUT/'DIAG10_0.npz',allow_pickle=True))
    lin=R.make_pipeline(R.SimpleImputer(strategy='median',keep_empty_features=True),R.StandardScaler(),R.Ridge(alpha=100.));lin.fit(tr[R.PHYSICS_COLUMNS],tr.sub_temp.to_numpy(),ridge__sample_weight=w[tm]);btr,bva=lin.predict(tr[R.PHYSICS_COLUMNS]),lin.predict(va[R.PHYSICS_COLUMNS]);y=tr.sub_temp.to_numpy()-btr
    imp=R.SimpleImputer(strategy='median',keep_empty_features=True).fit(tr[R.FEATURE_COLUMNS]);xtr,xva=imp.transform(tr[R.FEATURE_COLUMNS]),imp.transform(va[R.FEATURE_COLUMNS])
    models={'CB':R.CatBoostRegressor(iterations=600,depth=5,learning_rate=.04,l2_leaf_reg=10,random_strength=1,bootstrap_type='Bayesian',bagging_temperature=1,loss_function='RMSE',random_seed=726,thread_count=4,task_type='CPU',verbose=False,allow_writing_files=False),'ET':R.ExtraTreesRegressor(n_estimators=300,min_samples_leaf=20,max_features=.8,n_jobs=4,random_state=726)}
    report=[]
    for tag,m in models.items():
        m.fit(xtr,y,sample_weight=w[tm]);p=bva+m.predict(xva);diff=float(R.np.max(R.np.abs(p-cache[f'{tag}_726'])));assert diff<1e-10
        train=R.rmse(btr+m.predict(xtr),tr.sub_temp.to_numpy());val=R.rmse(p,va.sub_temp.to_numpy());assert abs(train-float(cache[f'{tag}_train_726']))<1e-10
        report.append(dict(member=tag,seed=726,fold=0,maxdiff=diff,train_rmse=train,validation_rmse=val))
    out=dict(status='PASS',scope='DIAG10 fold0 seed726 only, raw features/model retrain',checks=report)
    (R.HERE/'replay_verification.json').write_text(R.json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(R.json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
