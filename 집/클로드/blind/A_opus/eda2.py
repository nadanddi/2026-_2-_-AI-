import sys; sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot, os, pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",400)
D=boot.DATA
X=pd.read_csv(os.path.join(D,"train_X.csv")); y=pd.read_csv(os.path.join(D,"train_y.csv")); T=pd.read_csv(os.path.join(D,"test_X.csv"))
for d in (X,y,T):
    p=d.row_id.str.split("_",expand=True); d["gh"]=p[0]; d["day"]=p[1].astype(int); d["hr"]=p[2].astype(int)
for g in ["F13","F47"]:
    tr=set(X[X.gh==g].day); te=set(T[T.gh==g].day); ly=set(y[(y.gh==g)].day)
    s=""
    for d in range(min(tr|te),max(tr|te)+1):
        s+= "T" if d in te else ("L" if d in ly else ("x" if d in tr else "."))
    print(g, min(tr|te), s)
    # hours per day
    print(" test hours/day", T[T.gh==g].groupby("day").size().value_counts().to_dict(), " train hours/day", X[X.gh==g].groupby("day").size().value_counts().to_dict())
# per-gh column availability
cols=[c for c in X.columns if c not in("row_id","gh","day","hr")]
av=X.groupby("gh")[cols].apply(lambda d:d.notna().mean()).round(2)
print(av[(av[["out_temp","act_vent","act_side"]]>0).any(axis=1)])
# EC distribution in F13/F47
Y=y.merge(X,on=["row_id","gh","day","hr"])
for g in ["F13","F47"]:
    d=Y[Y.gh==g].sort_values(["day","hr"])
    print(g, d.groupby(d.day//10*10)[["sub_temp","sub_ec","in_temp"]].mean().round(3).T)
