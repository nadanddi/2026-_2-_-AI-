import pickle, numpy as np
R={}
for f in ["res_temp2","res_ec1","res_ec2","res_ec4","res_5","res_temp3"]:
    R.update(pickle.load(open(f+".pkl","rb")))
def cat(k): o=R[k][1]; return np.concatenate([v[0] for v in o.values()]),np.concatenate([v[1] for v in o.values()])
def daily(k):  # per-day squared-error sums for day-bootstrap
    y,p=cat(k); return ((y-p)**2).reshape(-1,24).sum(1), y
rng=np.random.RandomState(0)
def ci(ka,kb=None,pa=None,pb=None):
    y,_=cat(ka)
    a=pa if pa is not None else cat(ka)[1]; b=pb if pb is not None else (cat(kb)[1] if kb else None)
    ea=((y-a)**2).reshape(-1,24).sum(1); n=len(ea)
    B=rng.randint(0,n,(2000,n))
    ra=np.sqrt(ea[B].sum(1)/(24*n))
    s=f"rmse {np.sqrt(ea.sum()/(24*n)):.4f} [95% day-bootstrap {np.percentile(ra,2.5):.4f},{np.percentile(ra,97.5):.4f}]"
    if b is not None:
        eb=((y-b)**2).reshape(-1,24).sum(1); rb=np.sqrt(eb[B].sum(1)/(24*n)); d=ra-rb
        s+=f" | diff vs ref {np.sqrt(ea.sum()/(24*n))-np.sqrt(eb.sum()/(24*n)):+.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]"
    return s
y,_=cat("ec_notwin"); print("EC mean-baseline check n rows",len(y))
for k in ["ec_notwin","nw","nw_leaves7_n1500","ec_nw_l7_n1500_twin"]: print("EC",k,ci(k,"ec_notwin"))
pb=(cat("nw_leaves7_n1500")[1]+cat("ec_nw_l7_n1500_twin")[1])/2
print("EC blend l7+l7twin",ci("nw_leaves7_n1500",pa=pb,pb=cat("ec_notwin")[1]))
for k in ["base_twin","t_l7_n1500","t_l63_n600","l1","allgh_w5"]: print("T",k,ci(k,"base_twin"))
tb=(cat("base_twin")[1]+cat("t_l7_n1500")[1]+cat("t_l63_n600")[1])/3
print("T blend3",ci("base_twin",pa=tb,pb=cat("base_twin")[1]))
tb2=(cat("t_l7_n1500")[1]+cat("allgh_w5")[1])/2
print("T blend l7+allgh",ci("base_twin",pa=tb2,pb=cat("base_twin")[1]))
# temp worst days
y,p=cat("base_twin"); e=((y-p).reshape(-1,24)); b=e.mean(1); print("temp days with |bias|>2:", np.sum(np.abs(b)>2), "of", len(b))
keep=np.repeat(np.abs(b)<=2,24); print("temp base_twin rmse excluding those days:", np.sqrt(np.mean((y-p)[keep]**2)).round(4))
for k in ["t_l7_n1500","t_l63_n600"]:
    y2,p2=cat(k); print("   ",k, np.sqrt(np.mean((y2-p2)[keep]**2)).round(4))
print("   blend3", np.sqrt(np.mean((y-tb)[keep]**2)).round(4))
