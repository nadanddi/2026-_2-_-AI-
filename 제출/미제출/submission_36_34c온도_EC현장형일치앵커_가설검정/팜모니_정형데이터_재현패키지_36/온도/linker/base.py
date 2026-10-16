# package copy of the research linker (scratch linker/base.py); paths via LINKER_DATA / LINKER_WORK
import os; os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, pandas as pd, pickle
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.environ.get("LINKER_DATA", os.path.join(HERE, "..", "data"))   # original public train_X / train_y / test_X
S=os.environ.get("LINKER_WORK", os.path.join(HERE, "work")); os.makedirs(S, exist_ok=True)
O=["out_temp","out_hum","out_rad","out_wspd"]
IN=["in_temp","in_hum","in_co2"]; ACT=["act_vent","act_shade","act_thermal","act_heating","act_circfan"]
def build():
    tr=pd.read_csv(D+r"\train_X.csv"); te=pd.read_csv(D+r"\test_X.csv"); y=pd.read_csv(D+r"\train_y.csv")
    tr=tr.merge(y,on="row_id",how="left"); tr["split"]="train"; te["split"]="test"
    a=pd.concat([tr,te]); p=a.row_id.str.split("_",expand=True); a["farm"]=p[0]; a["day"]=p[1].astype(int); a["h"]=p[2].astype(int)
    a=a[a.farm.isin(["F13","F47"])].sort_values(["farm","day","h"])
    recs=a.groupby(["farm","day"]).agg(split=("split","first"),n=("h","size")).reset_index()
    def arr(cols):
        out=np.full((len(recs),24,len(cols)),np.nan)
        for i,(f,d) in enumerate(zip(recs.farm,recs.day)):
            m=a[(a.farm==f)&(a.day==d)]
            out[i,m.h.values]=m[cols].values.astype(float)
        return out
    R=dict(recs=recs,OUT=arr(O),IN=arr(IN),ACT=arr(ACT),EC=arr(["sub_ec"])[...,0],ST=arr(["sub_temp"])[...,0])
    # date groups: exact equality of 24h outdoor (rounded .1, NaN-tolerant via string)
    key=[tuple(np.round(R["OUT"][i],1).ravel()) for i in range(len(recs))]
    gid={};G=[]
    for k in key:
        ks=str(k); gid.setdefault(ks,len(gid)); G.append(gid[ks])
    recs["G"]=G
    pickle.dump(R,open(S+r"\base.pkl","wb"))
    return R
if __name__=="__main__":
    R=build(); r=R["recs"]
    print(r.groupby(["farm","split"]).size()); print("groups",r.G.nunique()); print(r.groupby("G").size().value_counts())
    print(r.n.value_counts())
    for f in ["F13","F47"]:
        s=r[r.farm==f]; print(f, s.day.min(), s.day.max(), "missing", sorted(set(range(s.day.min(),s.day.max()+1))-set(s.day)))
        print(" test", s[s.split=="test"].day.tolist())
