"""Final pipeline: causal features -> LightGBM ensembles trained on F13/F47 labelled train rows -> test predictions.
Also runs a causality check: perturbing later test inputs must not change earlier predictions."""
from features import *
import lightgbm as lgb, sys
P=dict(objective="regression",learning_rate=0.03,num_leaves=31,min_child_samples=40,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1)
DROP={"row_id","is_test","sub_temp","sub_ec","gh","t","day","twin_day","twin_it"}
TEMP_CFG=[dict(n=800,p={}),dict(n=1500,p=dict(num_leaves=7)),dict(n=600,p=dict(num_leaves=63))]
EC_CFG=[dict(n=1500,p=dict(num_leaves=7),twin=False),dict(n=1500,p=dict(num_leaves=7),twin=True)]
SEEDS=(0,1,2,3,4)

def masked_inputs(A):
    """copy of A where all test-row inputs are blanked: training-row features can then never depend on test inputs"""
    Am=A.copy(); Am.loc[Am.is_test==1,FEATS]=np.nan; return Am

def pipeline(A):
    F=build(A,["F13","F47"]); F=add_twin(F,A)
    Am=masked_inputs(A); Fm=build(Am,["F13","F47"]); Fm=add_twin(Fm,Am)
    wcols=[c for c in F.columns if c.startswith("out_") or c.startswith("ot_") or c.startswith("orad")]
    tr=Fm[Fm.is_test==0]; te=F[F.is_test==1].copy()
    # temperature
    cols=[c for c in F.columns if c not in DROP]
    d=tr[tr.sub_temp.notna()]; preds=[]
    for c in TEMP_CFG:
        for s in SEEDS:
            m=lgb.train({**P,**c["p"],"seed":s},lgb.Dataset(d[cols],d.sub_temp),c["n"]); preds.append(m.predict(te[cols]))
    te["sub_temp"]=np.mean(preds,0)
    # EC
    d=tr[tr.sub_ec.notna()]; preds=[]
    for c in EC_CFG:
        ccols=[x for x in cols if x not in wcols and (c["twin"] or not x.startswith("twin"))]
        for s in SEEDS:
            m=lgb.train({**P,**c["p"],"seed":s},lgb.Dataset(d[ccols],d.sub_ec),c["n"]); preds.append(m.predict(te[ccols]))
    te["sub_ec"]=np.mean(preds,0)
    return te[["row_id","gh","day","hr","sub_temp","sub_ec"]]

if __name__=="__main__":
    A,S=load()
    out=pipeline(A)
    sub=S[["row_id"]].merge(out[["row_id","sub_temp","sub_ec"]],on="row_id",how="left")
    assert len(sub)==1440 and sub.row_id.tolist()==S.row_id.tolist() and np.isfinite(sub[["sub_temp","sub_ec"]].values).all()
    sub.to_csv("pred_test.csv",index=False)
    print(sub.describe()); print(out.groupby("gh")[["sub_temp","sub_ec"]].mean())
    if "--check" in sys.argv:
        CUT=220
        A2=A.copy(); rng=np.random.RandomState(1)
        msk=(A2.is_test==1)&(A2.day>=CUT)
        for v in FEATS:
            if A2.loc[msk,v].notna().any(): A2.loc[msk,v]=A2.loc[msk,v].values*rng.uniform(0.5,1.5,msk.sum())+rng.normal(0,3,msk.sum())
        out2=pipeline(A2)
        m=out.merge(out2,on="row_id",suffixes=("","_2"))
        early=m.day<CUT
        print("causality check: max |diff| rows before day",CUT,":",np.abs(m.loc[early,["sub_temp","sub_ec"]].values-m.loc[early,["sub_temp_2","sub_ec_2"]].values).max(),
              "| rows after (should differ):",np.abs(m.loc[~early,"sub_temp"]-m.loc[~early,"sub_temp_2"]).max())
