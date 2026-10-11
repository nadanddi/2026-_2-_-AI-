# Candidate-consensus anchor (2026-10-12 집 클로드): for every mirror-CV target (v7 geometry & weights), keep the FULL candidate
# distribution; among LABELLED candidates compute mass m = sum p, mixture anchor A_mix = sum p*A / m (temp 23 h, EC 23 h) and
# dispersion s = sqrt(sum p*(A - A_mix)^2 / m).  Output per record for later rule tests (descriptive first).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v8")
from feat8 import *
import env
OUT=os.path.join(env.LOCAL,"linker_v8"); TRG=train_groups7(); rows=[]
for f in ["F13","F47"]:
    g="F47" if f=="F13" else "F13"; sh=-2 if f=="F13" else 2
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=[i for i in np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0] if TRAIN[i]]
        if not blk: continue
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}; m=s+sh
        oth=set(np.where((FARM==g)&(NUM>=m-1)&(NUM<=m+10))[0].tolist()); rem2=rem|oth
        touch=((FARM==f)&(NUM>=s-1)&(NUM<=s+10)); touch[list(oth)]=True
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False
        for b in blk:
            df,_=features7(b,0,lab,removed=rem2)
            if len(df)==0: continue
            sc=df[FEATS7].values.astype(float)@w; p=np.exp(sc-sc.max()); p/=p.sum()
            a=df.a.values.astype(int); L_=lab[a]; k=int(np.argmax(p))
            d=dict(rec=rid(b),top=rid(a[k]),p_top=float(p[k]),top_lab=bool(L_[k]),m=float(p[L_].sum()))
            for nm,AR in (("st",ST),("ec",EC)):
                if L_.any():
                    pa=p[L_]; Av=AR[a[L_],23]; ok=np.isfinite(Av); pa=pa[ok]; Av=Av[ok]
                    if pa.sum()>0:
                        mu=float((pa*Av).sum()/pa.sum()); d[f"A_{nm}"]=mu; d[f"s_{nm}"]=float(np.sqrt((pa*(Av-mu)**2).sum()/pa.sum()))
            rows.append(d)
        print(f,s,flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT,"mix_anchor_cv_v7geom.csv"),index=False)
