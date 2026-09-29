import sys; sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot, os, pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",200)
D=boot.DATA
X=pd.read_csv(os.path.join(D,"train_X.csv")); y=pd.read_csv(os.path.join(D,"train_y.csv")); T=pd.read_csv(os.path.join(D,"test_X.csv")); S=pd.read_csv(os.path.join(D,"sample_submission.csv"))
for d in (X,y,T):
    p=d.row_id.str.split("_",expand=True); d["gh"]=p[0]; d["day"]=p[1].astype(int); d["hr"]=p[2].astype(int)
print(X.head()); print(y.head()); print(T.head()); print(S.describe())
print("test gh", T.groupby("gh").agg(n=("day","size"),dmin=("day","min"),dmax=("day","max")))
print("test ids in train_X?", T.row_id.isin(X.row_id).sum())
g=X.groupby("gh").agg(n=("day","size"),dmin=("day","min"),dmax=("day","max"))
gy=y.groupby("gh").agg(ny=("day","size"),ymin=("day","min"),ymax=("day","max"),tna=("sub_temp",lambda s:s.isna().sum()),ena=("sub_ec",lambda s:s.isna().sum()),tm=("sub_temp","mean"),em=("sub_ec","mean"),es=("sub_ec","std"))
print(g.join(gy))
print("dup row ids", X.row_id.duplicated().sum())
print(X.isna().mean().round(3))
print(T.isna().mean().round(3))
print(X.describe().T); print(T.describe().T); print(y.describe().T)
