from pathlib import Path
import importlib.util
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('conditional_reference',H.parent/'temp_pipeline_20261003_v1/temp_pipeline_models.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p);m=p.m;np=m.np
GATE_COLS=['farm_id','sin','cos','in_temp','in_temp_mean','in_temp_std','in_temp_diff','delta','out_rad','act_vent','act_heating','in_hum']
_CACHE=None
def fit_fold(tr,va,w,seed,name,ct,phc):
    global _CACHE
    bs={726:7,727:101}[seed];ib=np.full(len(tr),np.nan);ic=np.full(len(tr),np.nan);audit=[]
    for k,fd in enumerate(p.inner_folds(tr)):
        tm,vm=m.common.split_mask(tr,fd);a,b=tr[tm].copy(),tr[vm].copy();m.TM.cold_v5.SEED=bs
        members=m.TM.temp_members(a,b,ct,phc,w[tm]);ib[vm]=.65*members['res']+.25*members['ridge']+.10*members['nys'];ic[vm]=m.TM.codex_fit_predict(a,b,w[tm],seed)
        audit.append(dict(fold=k,training_days=sorted(set(zip(a.farm,a.day.astype(int)))),query_days=sorted(set(zip(b.farm,b.day.astype(int)))),training_rows=len(a),query_rows=len(b)))
        print(f'gate inner {name}/{seed}/{k} fit {len(a)//24} days',flush=True)
    assert np.isfinite(ib).all() and np.isfinite(ic).all()
    def gate(d):return np.where(np.isnan(d.in_temp),1.,np.clip((d.in_temp.to_numpy()-8)/2,0,1))
    gt,gv=gate(tr),gate(va);beta=(.4+.1*gt)/(1-.3*gt);diff=ib-ic;target=tr.sub_temp.to_numpy()-beta*ib-(1-beta)*ic
    prep=m.make_pipeline(m.SimpleImputer(strategy='median',keep_empty_features=True),m.StandardScaler());xt=prep.fit_transform(tr[GATE_COLS]);xv=prep.transform(va[GATE_COLS]);xt=np.column_stack([np.ones(len(tr)),xt]);xv=np.column_stack([np.ones(len(va)),xv])
    reg=m.Ridge(alpha=100.,fit_intercept=False);design=xt*(gt*diff)[:,None];reg.fit(design,target,sample_weight=w)
    dt=gt*np.clip(xt@reg.coef_,-.15,.15);dv=gv*np.clip(xv@reg.coef_,-.15,.15)
    if _CACHE is None:
        z=dict(np.load(Path(m.env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True));_CACHE=(z,{str(r):i for i,r in enumerate(z['row_id'])})
    z,zi=_CACHE;ix=[zi[q] for q in va.row_id];base=z[f'{name}__MASK__{bs}'][ix];codex=z[f'{name}__CODEX__{seed}'][ix];pool=1-.3*gv;wb=.4+.1*gv+pool*dv;wc=.6-.4*gv-pool*dv;wp=.3*gv
    assert np.all(wb>=0) and np.all(wc>=0) and np.all(wp>=0) and np.max(np.abs(wb+wc+wp-1))<1e-12
    effective=codex+pool*dv*(base-codex)/(.6-.4*gv)
    before=beta*ib+(1-beta)*ic;after=before+dt*diff
    result={'G1':effective,'outer_BASE':base,'outer_CODEX':codex,'w_BASE':wb,'w_CODEX':wc,'w_PFN':wp,'gate_delta':dv,'cal_row_id':tr.row_id.to_numpy(dtype=str),'cal_y':tr.sub_temp.to_numpy(),'cal_w':w,'cal_BASE':ib,'cal_CODEX':ic,'cal_design':design,'cal_target':target,'cal_delta':dt}
    stats={'G1':dict(training_rows=len(tr),training_days=len(tr[['farm','day']].drop_duplicates()),inner_pool_before_rmse=m.rmse(before,tr.sub_temp.to_numpy()),inner_pool_after_rmse=m.rmse(after,tr.sub_temp.to_numpy()),weight_BASE_min=float(wb.min()),weight_BASE_max=float(wb.max()),weight_CODEX_min=float(wc.min()),weight_CODEX_max=float(wc.max()),gate_coef=reg.coef_.tolist(),inner_split_audit=audit)}
    return result,stats
