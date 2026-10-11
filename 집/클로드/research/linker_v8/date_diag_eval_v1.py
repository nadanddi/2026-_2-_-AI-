# Date-identification diagnostic for the 60 evaluation records (linker v7 visibility rules, decision hour 0 and 23).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v8")
from feat8 import *
import env
P=pd.read_csv(os.path.join(env.LOCAL,"linker_v7","eval60_predecessor_v7.csv")).set_index("row")
ev=[i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13","F47")]
rows=[]
for b in ev:
    d={}
    for h in (0,23):
        df,info=features7(b,h,TRAIN.copy(),removed=set())
        d[f"nmatch{h}"]=info.get("nmatch",np.nan) if info else np.nan; d[f"bpos{h}"]=info.get("bpos",np.nan) if info else np.nan
        d[f"ncand_isprev{h}"]=int((df.isprev==1).sum()) if len(df) else 0
        d[f"nlab_isprev{h}"]=int(((df.isprev==1)&TRAIN[df.a.values.astype(int)]).sum()) if len(df) else 0
    rows.append(dict(rec=rid(b),**d))
D=pd.DataFrame(rows).set_index("rec").join(P[["p_h0","prev_is_labelled"]])
pd.set_option("display.width",200); print(D.to_string())
print("date known at h0:",D.bpos0.notna().sum(),"/",len(D),"| at h23:",D.bpos23.notna().sum())
print(D.groupby(D.bpos0.notna()).agg(n=("p_h0","size"),conf=("p_h0",lambda s:(s>=.8).mean())))
