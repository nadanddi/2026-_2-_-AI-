# Descriptive trace (2026-10-12 집 클로드): date components (full-day outdoor >= 90% equal) of pass-2 records whose date has NO
# pass-1 record; order components by outdoor midnight continuity; list which records (train/eval, farm) sit on each date and the
# label-certain truth links touching them.  Descriptive map only (evaluation inputs used for the map, not for any rule).
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v8")
from feat8 import *
Xf=X.reshape(N,-1); par=list(range(N))
def fd(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for i in range(N):
    eq=(np.abs(Xf[i+1:]-Xf[i])<1e-6).mean(1)
    for j in np.where(eq>=.9)[0]+i+1: par[fd(j)]=fd(i)
comp=np.array([fd(i) for i in range(N)])
TLa=pd.read_pickle(S3+r"\truth.pkl"); pred={int(b):(int(a),bool(c)) for a,b,c in zip(TLa.a1,TLa.b,TLa.cert)}
succ={}
for b,(a,c) in pred.items(): succ.setdefault(a,[]).append((b,c))
has_p1={c for c in set(comp) if any(NUM[i]<179 for i in np.where(comp==c)[0])}
p2c=sorted({comp[i] for i in range(N) if NUM[i]>=179 and comp[i] not in has_p1}, key=lambda c:min(NUM[np.where(comp==c)[0]]))
# midnight continuity cost between components (use any member)
def cost(c1,c2):
    a=np.where(comp==c1)[0][0]; b=np.where(comp==c2)[0][0]
    pa=X[a,23]+0.5*(X[a,23]-X[a,22]); return float(np.sum(np.array([1,1,.3,.5])*np.abs(pa-X[b,0])/sdO))
allc=sorted(set(comp))
for c in p2c:
    mem=np.where(comp==c)[0]
    desc=" ".join(f"{rid(i)}{'' if TRAIN[i] else '*'}" for i in sorted(mem,key=lambda i:(FARM[i],NUM[i])))
    cs=sorted(((cost(c2,c),c2) for c2 in allc if c2!=c))[:3]
    prevs=" / ".join("%.2f:%s%s"%(v,"P1date " if c2 in has_p1 else "", ",".join(rid(i)+('' if TRAIN[i] else '*') for i in np.where(comp==c2)[0])) for v,c2 in cs)
    links=[]
    for i in mem:
        if i in pred: a,cc=pred[i]; links.append(f"{rid(i)}<-{rid(a)}{'(c)' if cc else ''}")
    print(f"DATE[{desc}]\n   best previous-date components (cost:members): {prevs}\n   truth links: {' '.join(links)}")
