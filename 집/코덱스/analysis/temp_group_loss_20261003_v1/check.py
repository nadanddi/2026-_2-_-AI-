from pathlib import Path
import json
from model import m,np,loss_parts
H=Path(__file__).resolve().parent
rng=np.random.default_rng(20261003);e=rng.normal(size=12);w=rng.uniform(.2,1,size=12);groups=np.repeat(np.arange(3),4);checks=[]
for kind in ['L1','S1','D1','ZERO']:
    loss,g,h=loss_parts(e,w,groups,kind);numeric=[];hd=[]
    for i in range(len(e)):
        a=e.copy();b=e.copy();a[i]+=1e-5;b[i]-=1e-5
        la,ga,_=loss_parts(a,w,groups,kind);lb,gb,_=loss_parts(b,w,groups,kind)
        numeric.append((la-lb)/2e-5);hd.append((ga[i]-gb[i])/2e-5)
    gd=float(np.max(np.abs(g-numeric)));hdiff=float(np.max(np.abs(h-hd)));assert gd<1e-8 and hdiff<1e-8 and np.all(h>0)
    checks.append(dict(kind=kind,gradient_maxdiff=gd,hessian_diagonal_maxdiff=hdiff))
m.common.load_raw=m.TM.masked_loader
try:lab,ct,phc=m.TM.build_world()
finally:m.common.load_raw=m.TM.ORIG;m.harness._CACHE.clear()
tx,_,sx=m.TM.masked_loader();cf=m.build_features(tx,sx).set_index('row_id')
for c in m.FEATURE_COLUMNS:
    if c not in lab:lab[c]=cf.loc[lab.row_id,c].to_numpy()
weights=m.TF.row_weights(lab,.2,w_noisy=.2);tm,vm=m.common.split_mask(lab,m.diag_folds(lab)[0]);tr,va=lab[tm],lab[vm];w=weights[tm]
lin=m.make_pipeline(m.SimpleImputer(strategy='median',keep_empty_features=True),m.StandardScaler(),m.Ridge(alpha=100.));lin.fit(tr[m.PHYSICS_COLUMNS],tr.sub_temp.to_numpy(),ridge__sample_weight=w)
r=tr.sub_temp.to_numpy()-lin.predict(tr[m.PHYSICS_COLUMNS]);offset=np.average(r,weights=w);groups=m.pd.factorize(m.pd.MultiIndex.from_frame(tr[['farm','day']]),sort=False)[0]
def objective(y,p):return loss_parts(p-y,w,groups,'ZERO')[1:]
model=m.LGBMRegressor(objective=objective,n_estimators=220,learning_rate=.035,num_leaves=12,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=726,boost_from_average=False)
model.fit(tr[m.FEATURE_COLUMNS],r-offset);p=lin.predict(va[m.PHYSICS_COLUMNS])+offset+model.predict(va[m.FEATURE_COLUMNS]);ref=m.TM.codex_fit_predict(tr,va,w,726)
delta=float(np.max(np.abs(p-ref)));rmse_delta=m.rmse(p,va.sub_temp.to_numpy())-m.rmse(ref,va.sub_temp.to_numpy())
# Weighted built-in float32 labels may cause tiny numeric deviations; anything substantial blocks full run.
assert delta<1e-5
answer=dict(status='PASS',objective_finite_differences=checks,zero_vs_original_maxdiff=delta,zero_vs_original_rmse_delta=rmse_delta)
(H/'preflight.json').write_text(json.dumps(answer,indent=2),encoding='utf-8');print(json.dumps(answer))
