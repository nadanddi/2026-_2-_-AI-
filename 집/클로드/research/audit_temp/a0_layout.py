import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np, os
print(env.DATA)
tX=pd.read_csv(os.path.join(env.DATA,"train_X.csv")); sX=pd.read_csv(os.path.join(env.DATA,"test_X.csv")); ty=pd.read_csv(os.path.join(env.DATA,"train_y.csv"))
for d in (tX,sX):
    d["farm"]=d.row_id.str[:3]; d["day"]=d.row_id.str[4:7].astype(int); d["hour"]=d.row_id.str[8:10].astype(int)
for f in ["F13","F47"]:
    tr=sorted(tX[tX.farm==f].day.unique()); te=sorted(sX[sX.farm==f].day.unique())
    allp=set(tr)|set(te); lo,hi=min(allp),max(allp)
    miss=[d for d in range(lo,hi+1) if d not in allp]
    print(f,"train",tr[0],tr[-1],len(tr),"test",te[0],te[-1],len(te),"missing",miss)
    print(" test days",te)
    # hours per day completeness
    print(" train rows/day uniq", tX[tX.farm==f].groupby('day').size().value_counts().to_dict())
print("train NaN frac F13/F47:", tX[tX.farm.isin(['F13','F47'])].drop(columns=['row_id','farm','day','hour']).isna().mean().round(4).to_dict())
print("test NaN frac:", sX.drop(columns=['row_id','farm','day','hour']).isna().mean().round(4).to_dict())
print("farms in train", tX.farm.unique()[:20], len(tX.farm.unique()))
