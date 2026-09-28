from common import *
R=pd.read_pickle("daykeys.pkl")
gid={k:i for i,k in enumerate(R.ot.unique())}; R["gid"]=R.ot.map(gid)
A,S=load()
A=A.merge(R[["gh","day","gid"]],on=["gh","day"],how="left")
W=A[A.gh.isin(["F13","F47"])&A.gid.notna()].groupby(["gid","hr"])[["out_temp","out_hum","out_rad","out_wspd"]].first().reset_index()
P={c:W.pivot(index="gid",columns="hr",values=c) for c in ["out_temp","out_hum","out_rad","out_wspd"]}
G=P["out_temp"].index.values
# extrapolated end: predict hr0 of next day from hr22,hr23
ot=P["out_temp"].values; oh=P["out_hum"].values
pred0=ot[:,23]+(ot[:,23]-ot[:,22])*0.5
cost=np.abs(pred0[:,None]-ot[None,:,0]) + 0.1*np.abs(oh[:,23][:,None]-oh[None,:,0])
np.fill_diagonal(cost,99)
# check known consecutive pairs in F13 first half
f13=R[(R.gh=="F13")&(R.day<179)].sort_values("day").drop_duplicates("gid").gid.astype(int).values
idx={g:i for i,g in enumerate(G)}
ranks=[]
for a,b in zip(f13[:-1],f13[1:]):
    c=cost[idx[a]]; ranks.append((c<c[idx[b]]).sum())
ranks=np.array(ranks); print("rank of true successor among all dates: ", np.bincount(np.minimum(ranks,10)))
best=np.argsort(cost,axis=1)
for g in list(range(113,136))+[77,78,79,80,81,82]:
    i=idx[g]; print(g, "best succ:", [(int(G[j]),round(cost[i,j],2)) for j in best[i,:3]], " best pred:", [(int(G[j]),round(cost[j,i],2)) for j in np.argsort(cost[:,i])[:3]])
