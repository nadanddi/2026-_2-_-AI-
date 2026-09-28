from features import *
import lightgbm as lgb
A,S=load()
FOLDS=s2_folds(A)
F_ALL=pd.read_pickle("feat_all.pkl")
DROP={"row_id","is_test","sub_temp","sub_ec","gh","t","day","twin_day","twin_it"}
def feats(F, twin=True, extra_drop=()):
    return [c for c in F.columns if c not in DROP and c not in extra_drop and (twin or not c.startswith("twin"))]
P=dict(objective="regression",learning_rate=0.03,num_leaves=31,min_child_samples=40,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1)
def run(target, ghs=("F13","F47"), twin=True, seeds=(0,1,2), n=800, params=None, extra_drop=(), resid=None, folds=None, weight=None, trfilter=None):
    """returns per-fold rmse matrix [seed x fold] and oof preds"""
    params={**P,**(params or {})}
    F=F_ALL[F_ALL.gh.isin(ghs)]
    out=[]; oof={}
    for fi,(g,B) in enumerate(folds or FOLDS):
        hidden={(g,d) for d in B}
        Ff=add_twin(F,A,hidden) if twin else F
        cols=feats(Ff,twin,extra_drop)
        isval=(Ff.gh==g)&Ff.day.isin(B)
        tr=Ff[~isval & Ff[target].notna() & (Ff.is_test==0)]
        if trfilter is not None: tr=tr[trfilter(tr,g,B)]
        va=Ff[isval & Ff[target].notna()]
        ytr=tr[target]-(tr[resid] if resid else 0)
        row=[]
        preds=[]
        for s in seeds:
            m=lgb.train({**params,"seed":s},lgb.Dataset(tr[cols],ytr,weight=None if weight is None else weight(tr)),n)
            p=m.predict(va[cols])+(va[resid].values if resid else 0)
            preds.append(p); row.append(np.sqrt(np.mean((p-va[target].values)**2)))
        out.append(row); oof[(g,B[0])]=(va[target].values,np.mean(preds,0))
    return np.array(out).T, oof
def summarize(name, M, oof):
    yy=np.concatenate([v[0] for v in oof.values()]); pp=np.concatenate([v[1] for v in oof.values()])
    pooled=np.sqrt(np.mean((yy-pp)**2))
    per_seed_pooled=[]
    print(f"{name:28s} pooled(seed-avg)={pooled:.4f} | folds:", " ".join(f"{x:.3f}" for x in M.mean(0)), "| seed-sd of fold-mean %.4f"%M.mean(1).std())
    return pooled
