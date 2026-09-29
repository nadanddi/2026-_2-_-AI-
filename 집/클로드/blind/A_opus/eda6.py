from common import *
from collections import defaultdict
pd.set_option("display.width",250)
A,S=load()
B=A[A.out_temp.notna()]
print(B.gh.unique())
rows=[]
for (g,d),x in B.groupby(["gh","day"]):
    x=x.sort_values("hr")
    if len(x)<24: continue
    rows.append((g,d,tuple(x.out_temp.round(1)),tuple(x.out_hum.round(0)) if x.out_hum.notna().all() else None, x.is_test.max(), x.sub_ec.mean(), x.sub_temp.mean(), x.in_temp.mean()))
R=pd.DataFrame(rows,columns=["gh","day","ot","oh","test","ec","st","it"])
grp=R.groupby("ot")
R["nkey"]=grp.gh.transform("size")
print("group size distribution:", R.drop_duplicates("ot").ot.map(R.ot.value_counts()).value_counts().to_dict())
comp=R.groupby("ot").apply(lambda x: "+".join(sorted(x.gh+("t" if False else "")))).value_counts()
print(comp.head(20))
# for F47 and F13, print day sequence with group ids
gid={k:i for i,k in enumerate(R.ot.unique())}
R["gid"]=R.ot.map(gid)
for g in ["F13","F47","F32"]:
    x=R[R.gh==g].sort_values("day")
    print(g, " ".join(f"{d}:{i}{'T' if t else ''}" for d,i,t in zip(x.day,x.gid,x.test)))
R.to_pickle("daykeys.pkl")
