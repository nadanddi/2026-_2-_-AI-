from common import *
A,S=load()
folds=s2_folds(A)
H=A[A.gh.isin(["F13","F47"])]
ECd=H.groupby(["gh","day"]).sub_ec.mean()
ECh=H.set_index(["gh","day","hr"]).sub_ec
def other_level(g,tw,allowed,thr=0.25,win=4):
    etw=ECd[(g,tw)]
    nb=[d for d in range(tw-win,tw+win+1) if d!=tw and d in allowed]
    if not nb: return etw, etw
    e=np.array([ECd[(g,d)] for d in nb])
    far=e[np.abs(e-etw)>thr*etw]
    return (np.median(far) if len(far) else etw), etw
res=[]
for g,B in folds:
    T=day_table(A,g).set_index("day")
    lab=[d for d in T.index if T.loc[d,"test"]==0 and not np.isnan(ECd.get((g,d),np.nan))]
    allowed=set(d for d in lab if d<B[0])
    prev=sorted(allowed)
    e1=ECd[(g,prev[-1])]; e2=np.mean([ECd[(g,d)] for d in prev[-2:]]); e4=np.mean([ECd[(g,d)] for d in prev[-4:]])
    for d in B:
        y=ECh[(g,d)].values
        tw=T.loc[d,"twin"]
        if not np.isnan(tw) and int(tw) in allowed:
            ol,etw=other_level(g,int(tw),allowed); has=1
        else: ol,etw,has=e2,np.nan,0
        res.append(dict(g=g,b=B[0],d=d,has=has,y=y,E0=np.full(24,ECd[g][list(allowed)].mean()),E1=np.full(24,e1),E2=np.full(24,e2),E4=np.full(24,e4),E3=np.full(24,ol),
                        Etw=np.full(24,etw if has else e2)))
R=pd.DataFrame(res)
def rmse(col,sub=None):
    r=R if sub is None else R[sub]
    return np.sqrt(np.mean(np.concatenate([(a-b)**2 for a,b in zip(r[col],r.y)])))
for c in ["E0","E1","E2","E4","E3","Etw"]:
    print(c, "all %.4f  twin-days %.4f  no-twin %.4f"%(rmse(c),rmse(c,R.has==1),rmse(c,R.has==0)))
print("per fold E2 vs E3:")
for (g,b),r in R.groupby(["g","b"]):
    f=lambda c: np.sqrt(np.mean(np.concatenate([(a-y)**2 for a,y in zip(r[c],r.y)])))
    print(g,b,len(r),"has",r.has.sum(), "E2 %.3f E3 %.3f E1 %.3f"%(f("E2"),f("E3"),f("E1")))
