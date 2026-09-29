"""Structural EC estimator: calendar position via same-gh earlier weather twin, two-compartment level context."""
from common import *

def cal_positions(T):
    """T: day_table. cal = calendar proxy expressed in first-half relative days, causal (uses earlier days only)."""
    cal={}; mode2=False; last=None
    for _,r in T.iterrows():
        d=r.day; tw=r.twin
        if tw==tw:
            tw=int(tw)
            if d-tw>6: mode2=True
            c=cal.get(tw,tw)
        else:
            c=last if (mode2 and last is not None) else d
        cal[d]=c; last=c
    return cal

def two_levels(vals):
    v=np.sort(np.asarray(vals))
    if len(v)<2: return (v[0],v[0]) if len(v) else (np.nan,np.nan)
    best=None
    for k in range(1,len(v)):
        a,b=v[:k],v[k:]; sse=((a-a.mean())**2).sum()+((b-b.mean())**2).sum()
        if best is None or sse<best[0]: best=(sse,np.median(a),np.median(b))
    return best[1],best[2]

def predict_days(A, g, targets, hidden, ECd, ECh, w=3, p_hi=None, shape=True, twin_mode="other"):
    """predict hourly EC for days `targets` of gh g. hidden: days whose labels are unavailable.
    returns dict day->24 preds, and info"""
    T=day_table(A,g); cal=cal_positions(T); Td=T.set_index("day")
    lab={d for d in T.day if (g,d) in ECd.index and ECd[(g,d)]==ECd[(g,d)] and d not in hidden}
    out={}; info={}
    for d in targets:
        c=cal[d]
        ctx=[e for e in lab if e<d and abs(cal[e]-c)<=w]
        if len(ctx)<2: ctx=sorted([e for e in lab if e<d],key=lambda e:abs(cal[e]-c))[:4]
        tw=Td.loc[d,"twin"]; tw=int(tw) if tw==tw else None
        vals=np.array([ECd[(g,e)] for e in ctx])
        lo,hi=two_levels(vals)
        if tw is not None and tw in lab and twin_mode!="none":
            et=ECd[(g,tw)]
            others=[e for e in ctx if e!=tw]
            ov=np.array([ECd[(g,e)] for e in others]) if others else np.array([et])
            if twin_mode=="other":
                # the other compartment = the level farther from twin
                lvl = hi if abs(et-lo)<abs(et-hi) else lo
                if abs(hi-lo) < 0.15*max(hi,1e-6): lvl=np.median(ov)
            else: lvl=et
            src=[e for e in others if abs(ECd[(g,e)]-lvl)<=abs(ECd[(g,e)]-et)] or others or [tw]
            has=1
        else:
            ph=0.5 if p_hi is None else p_hi.get(d,0.5)
            lvl=ph*hi+(1-ph)*lo; src=ctx; has=0
        if shape and src:
            prof=np.nanmean([ECh.loc[(g,e)].values/ECd[(g,e)] for e in src],axis=0)
            prof=prof/np.nanmean(prof)
            out[d]=lvl*prof
        else: out[d]=np.full(24,lvl)
        info[d]=dict(cal=c,has=has,lo=lo,hi=hi,lvl=lvl,n=len(ctx))
    return out,info

if __name__=="__main__":
    A,S=load(); folds=s2_folds(A)
    H=A[A.gh.isin(["F13","F47"])]
    ECd=H.groupby(["gh","day"]).sub_ec.mean(); ECh=H.set_index(["gh","day","hr"]).sub_ec
    for w in [2,3,4,6]:
      for shape in [False,True]:
        for tm in ["other","none"]:
            ys=[];ps=[];rows=[]
            for g,B in folds:
                pr,info=predict_days(A,g,B,set(B),ECd,ECh,w=w,shape=shape,twin_mode=tm)
                for d in B:
                    y=ECh.loc[(g,d)].values; ys.append(y); ps.append(pr[d]); rows.append((g,B[0],info[d]["has"],np.mean((y-pr[d])**2)))
            R=pd.DataFrame(rows,columns=["g","b","has","mse"])
            fold=R.groupby(["g","b"]).mse.mean()**.5
            print(f"w={w} shape={shape} twin={tm}: pooled {np.sqrt(R.mse.mean()):.4f} twin-days {np.sqrt(R[R.has==1].mse.mean()):.4f} no-twin {np.sqrt(R[R.has==0].mse.mean()):.4f} | folds "+" ".join(f"{x:.3f}" for x in fold.values))
