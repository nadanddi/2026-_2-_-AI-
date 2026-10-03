"""First-fold mathematical diagnostic; no validation scoring or new candidate."""
from pathlib import Path
import sys, json, math, importlib.util, gc
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('log_leaf_diagnostic_source', H/'run.py')
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def main():
    dest = H/'leaf_diagnostic_v1.json'
    assert not dest.exists()
    lab, core, wv, folds, _ = L.S.loadec()
    v,k,tm,vm = next(z for z in folds if z[0]=='DIAG10' and z[1]==0)
    tr,va = L.S.seasonal(lab[tm], lab[vm], wv)
    tr,va = tr.reset_index(drop=True),va.reset_index(drop=True)
    cols = [c for c in core.FULL if c!='day']+['season']
    b = L.baseline(tr); bt,bq = b(tr),b(va); y = tr.sub_ec.to_numpy()
    r = y/bt; model = core.et(7); model.steps[-1][1].n_jobs=2
    with threadpool_limits(limits=2): model.fit(tr[cols],np.log(r))
    model.steps[-1][1].n_jobs=1
    xt,xq = model.steps[0][1].transform(tr[cols]),model.steps[0][1].transform(va[cols])
    counts,values = L.leaf_arrays(model,xt,r)
    arithmetic,geometric = L.predict(model,va,cols,bq,counts,values)
    saved = pd.read_csv(L.OUT/'DIAG10_0_7.csv',float_precision='round_trip').set_index('row_id').reindex(va.row_id)
    diff = float(np.max(np.abs(arithmetic-saved.new_et_raw.to_numpy())))
    olddiff = float(np.max(np.abs(geometric-saved.old_log_raw.to_numpy())))
    assert diff<1e-12 and olddiff<1e-12
    stats = dict(occupied_leaves=0,multirow_leaves=0,varying_b_leaves=0,train_visits=0,multirow_train_visits=0,
                 query_visits=0,multirow_query_visits=0,varying_b_query_visits=0)
    weighted_raw = np.zeros(len(va)); orig_raw = np.zeros(len(va))
    loss_mean,loss_weighted = [],[]; manual=[]; maxleaf=0.
    for tree,n,m in zip(model.steps[-1][1].estimators_,counts,values):
        leaf = tree.apply(xt); qleaf = tree.apply(xq); good=n>0; multi=n>1
        low = np.full(len(n),np.inf); high=np.full(len(n),-np.inf)
        np.minimum.at(low,leaf,bt);np.maximum.at(high,leaf,bt)
        varying=good & ((high-low)>1e-10)
        by=np.bincount(leaf,weights=bt*y,minlength=len(n));bb=np.bincount(leaf,weights=bt*bt,minlength=len(n))
        w=np.divide(by,bb,out=np.zeros_like(by),where=good)
        yy=np.bincount(leaf,weights=y,minlength=len(n)); orig=np.divide(yy,n,out=np.zeros_like(yy),where=good)
        assert (n[qleaf]>0).all()
        assert np.max(np.abs(w[good]-m[good])[~varying[good]])<1e-10
        maxleaf=max(maxleaf,float(np.max(abs(w[good]-m[good]))))
        weighted_raw += w[qleaf]; orig_raw += orig[qleaf]
        la=(y-bt*m[leaf])**2;lw=(y-bt*w[leaf])**2
        assert np.all(np.bincount(leaf,weights=lw,minlength=len(n))[good] <= np.bincount(leaf,weights=la,minlength=len(n))[good]+1e-10)
        loss_mean.append(float(la.sum()));loss_weighted.append(float(lw.sum()))
        for name,inc in [('occupied_leaves',good.sum()),('multirow_leaves',multi.sum()),('varying_b_leaves',varying.sum()),
                         ('train_visits',len(tr)),('multirow_train_visits',n[multi].sum()),('query_visits',len(va)),
                         ('multirow_query_visits',multi[qleaf].sum()),('varying_b_query_visits',varying[qleaf].sum())]:
            stats[name]+=int(inc)
        if len(manual)<3:
            for j in np.flatnonzero(varying)[:3-len(manual)]:
                ix=np.flatnonzero(leaf==j)
                a=math.fsum(float(r[t]) for t in ix)/len(ix)
                z=math.fsum(float(bt[t])*float(y[t]) for t in ix)/math.fsum(float(bt[t])**2 for t in ix)
                assert abs(a-m[j])<1e-12 and abs(z-w[j])<1e-12
                manual.append(dict(n=len(ix),b_min=float(low[j]),b_max=float(high[j]),arithmetic_ratio=a,weighted_ratio=z))
    nt=len(counts);weighted_raw=bq*weighted_raw/nt;orig_raw=orig_raw/nt
    result=dict(status='PASS',scope='First DIAG0 seed7 fixed-partition diagnostic only; no validation score or candidate',
                stats=stats,query_multirow_fraction=stats['multirow_query_visits']/stats['query_visits'],
                query_varying_b_fraction=stats['varying_b_query_visits']/stats['query_visits'],
                stored_arithmetic_maxdiff=diff,stored_geometric_maxdiff=olddiff,
                max_leaf_ratio_change=maxleaf,weighted_vs_arithmetic_raw_maxdiff=float(np.max(abs(weighted_raw-arithmetic))),
                weighted_vs_arithmetic_raw_mean_absdiff=float(np.mean(abs(weighted_raw-arithmetic))),
                unnormalized_original_leaf_mean_vs_arithmetic_raw_maxdiff=float(np.max(abs(orig_raw-arithmetic))),
                summed_leaf_training_sse=dict(arithmetic_ratio=math.fsum(loss_mean),weighted_ratio=math.fsum(loss_weighted)),
                manual_leaf_checks=manual,source_sha256=L.S.sha(H/'run.py'))
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    del model;gc.collect()

if __name__=='__main__':main()
