from base import *
import collections
R=pickle.load(open(S+r"\base.pkl","rb")); r=R["recs"]; X=R["OUT"]; T=pd.read_pickle(S+r"\t1.pkl")
sd=np.nanstd(X[(r.split=="train").values].reshape(-1,4),0)   # rule fix 10-11: training records only
def oj(a,b):
    pa=X[a,23]+0.5*(X[a,23]-X[a,22]); pb=X[b,0]-0.5*(X[b,1]-X[b,0]); w=np.array([1,1,0.3,0.5])
    return float(np.sum(w*np.abs(pa-pb)/sd))
T["oj1"]=[oj(a,b) for a,b in zip(T.a1,T.b)]
T["cert"]=(T.c1<=1.0)&(T.c2-T.c1>=1.0)&(T.oj1<=1.2)
dup=T[T.cert].a1.value_counts(); T.loc[T.a1.isin(dup[dup>1].index),"cert"]=False
T["fb"]=r.farm.values[T.b]; T["nb"]=r.day.values[T.b]; T["fa"]=r.farm.values[T.a1]; T["na"]=r.day.values[T.a1]
T["segb"]=np.where(T.nb<179,1,2); T["sega"]=np.where(T.na<179,1,2)
C=T[T.cert]
print("certain",len(C)); print(pd.crosstab([C.segb,C.fb],[C.fa,C.sega]))
print("a number < b number frac (same farm)", (C[C.fb==C.fa].na<C[C.fb==C.fa].nb).mean())
print("cross-farm a num < b num frac",(C[C.fb!=C.fa].na<C[C.fb!=C.fa].nb).mean())
C2=C[(C.segb==2)]; print(C2[["fb","nb","fa","na","c1","c2","oj1"]].sort_values(["fb","nb"]).round(2).to_string())
T.to_pickle(S+r"\truth.pkl")
