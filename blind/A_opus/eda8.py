from common import *
R=pd.read_pickle("daykeys.pkl")
gid={k:i for i,k in enumerate(R.ot.unique())}; R["gid"]=R.ot.map(gid)
A,S=load()
A=A.merge(R[["gh","day","gid"]],on=["gh","day"],how="left")
cols=["out_temp","out_hum","out_rad","out_wspd","in_temp","in_hum","in_co2","act_vent","act_shade","act_thermal","act_heating","act_circfan","act_co2","act_fog","sub_temp","sub_ec"]
res=[]
for (g,k),x in A[A.gid.notna()].groupby(["gh","gid"]):
    ds=sorted(x.day.unique())
    if len(ds)!=2: continue
    a=x[x.day==ds[0]].set_index("hr")[cols]; b=x[x.day==ds[1]].set_index("hr")[cols]
    res.append(dict(gh=g, adj=ds[1]-ds[0]==1, **{c:(a[c]==b[c]).mean() for c in cols}))
res=pd.DataFrame(res); print(res.groupby(["gh","adj"]).mean().round(2).T)
# cross gh same date
res=[]
for k,x in A[A.gid.notna()].groupby("gid"):
    p=x.groupby(["gh","day"]).size().index.tolist()
    f13=[d for g,d in p if g=="F13"]; f47=[d for g,d in p if g=="F47"]
    for d1 in f13:
        for d2 in f47:
            a=x[(x.gh=="F13")&(x.day==d1)].set_index("hr")[cols]; b=x[(x.gh=="F47")&(x.day==d2)].set_index("hr")[cols]
            res.append({c:(a[c]==b[c]).mean() for c in cols})
print("F13 vs F47 same date"); print(pd.DataFrame(res).mean().round(2))
