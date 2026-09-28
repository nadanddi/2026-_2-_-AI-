import sys; sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot, os, pandas as pd, numpy as np
D=boot.DATA
FEATS=["out_temp","out_hum","out_rad","out_wspd","in_temp","in_hum","in_co2","in_rad","act_vent","act_side","act_shade","act_thermal","act_valve","act_heating","act_circfan","act_co2","act_fog","act_cool","act_pump"]
def _parse(d):
    p=d.row_id.str.split("_",expand=True); d["gh"]=p[0]; d["day"]=p[1].astype(int); d["hr"]=p[2].astype(int); d["t"]=d.day*24+d.hr
    return d
def load():
    X=_parse(pd.read_csv(os.path.join(D,"train_X.csv"))); y=pd.read_csv(os.path.join(D,"train_y.csv")); T=_parse(pd.read_csv(os.path.join(D,"test_X.csv")))
    S=pd.read_csv(os.path.join(D,"sample_submission.csv"))
    X["is_test"]=0; T["is_test"]=1
    A=pd.concat([X,T],ignore_index=True).merge(y,on="row_id",how="left")
    A=A.sort_values(["gh","t"]).reset_index(drop=True)
    return A,S
def grid(A, gh):
    """full hourly grid for one greenhouse; missing hours become NaN rows"""
    d=A[A.gh==gh].set_index("t")
    full=np.arange(d.index.min(), d.index.max()+1)
    return d.reindex(full)

def blocks(days):
    """split sorted day list into runs of consecutive days"""
    days=sorted(days); out=[[days[0]]]
    for d in days[1:]:
        if d==out[-1][-1]+1: out[-1].append(d)
        else: out.append([d])
    return out

def s2_folds(A):
    """validation folds: labelled blocks of F13/F47 adjacent to the test period (last run before first test block,
    runs between test blocks, run after last test block)"""
    folds=[]
    for g in ["F13","F47"]:
        d=A[A.gh==g]
        tfirst=d[d.is_test==1].day.min()
        lab=sorted(d[(d.is_test==0)&d.sub_ec.notna()].day.unique())
        runs=blocks(lab)
        for r in runs:
            if r[-1]<tfirst-1-1:  # the first-half run: take its last 5 days (they are 're-appearing' dates)
                continue
            folds.append((g,r))
        # first-half run which ends right before first test block
        last=[r for r in runs if r[-1]<tfirst][-1]
        folds=[f for f in folds if not (f[0]==g and f[1]==last)]
        folds.append((g,last[-5:]))
    return folds

def day_table(A, gh):
    """day-level info for one greenhouse: weather fingerprint and earlier same-gh twin day (same out_* profile)"""
    d=A[A.gh==gh]
    rows=[]
    for day,x in d.groupby("day"):
        x=x.sort_values("hr")
        key=tuple(np.round(x.out_temp.values,1)) if len(x)==24 and x.out_temp.notna().all() else None
        rows.append(dict(day=day,key=key,test=int(x.is_test.max()),lab=int(x.sub_ec.notna().any() or x.sub_temp.notna().any())))
    T=pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    seen={}; tw=[]
    for _,r in T.iterrows():
        k=r.key
        tw.append(seen.get(k,np.nan) if k is not None else np.nan)
        if k is not None and k not in seen: seen[k]=r.day
    T["twin"]=tw   # earliest earlier day with same weather fingerprint (same calendar date, other compartment)
    return T
