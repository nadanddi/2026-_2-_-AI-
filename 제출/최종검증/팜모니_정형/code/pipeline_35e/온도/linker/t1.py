from base import *
import collections
R=pickle.load(open(S+r"\base.pkl","rb")); r=R["recs"]; EC=R["EC"]; ST=R["ST"]; X=R["OUT"]
tr=np.where(r.split=="train")[0]
print("within-day hourly |dEC| at 22->23, 0->1 q",np.nanquantile(np.abs(EC[tr,23]-EC[tr,22]),[.5,.9,.99]).round(4),np.nanquantile(np.abs(EC[tr,1]-EC[tr,0]),[.5,.9,.99]).round(4))
print("within-day hourly |dST| q",np.nanquantile(np.abs(ST[tr,23]-ST[tr,22]),[.5,.9,.99]).round(3),np.nanquantile(np.abs(ST[tr,1]-ST[tr,0]),[.5,.9,.99]).round(3))
print("nan EC at 0/23:",np.isnan(EC[tr,0]).sum(),np.isnan(EC[tr,23]).sum())
# extrapolated continuity cost
def pe(A,i): return A[i,23]+0.5*(A[i,23]-A[i,22])
def pb(A,j): return A[j,0]-0.5*(A[j,1]-A[j,0])
rows=[]
for j in tr:
    ce=np.abs(pe(EC,tr)-pb(EC,j)); cs=np.abs(pe(ST,tr)-pb(ST,j))
    c=ce/0.01+cs/0.3; c[tr==j]=np.inf
    o=np.argsort(c); a1,a2=tr[o[0]],tr[o[1]]
    rows.append(dict(b=j,a1=a1,a2=a2,c1=c[o[0]],c2=c[o[1]],ce1=ce[o[0]],cs1=cs[o[0]],ce2=ce[o[1]],same_farm=r.farm[a1]==r.farm[j],sameG=r.G[a1]==r.G[j]))
T=pd.DataFrame(rows)
print(T[["c1","c2","ce1","cs1","ce2"]].describe(percentiles=[.25,.5,.75,.9]).round(3))
print("best a same farm frac",T.same_farm.mean(),"best a same date group (impossible)",T.sameG.mean())
T.to_pickle(S+r"\t1.pkl")
