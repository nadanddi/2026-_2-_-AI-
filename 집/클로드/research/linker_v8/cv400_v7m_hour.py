# MIRROR CV, linker v7 weights trained at hour 0 (as v7), decision at hour H (argv): records whose date is only matched later in the day (hours 0..H outdoor) may gain confidence; causal (row hour >= H uses inputs <= H).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v8")
from feat8 import *
import env
OUT=os.path.join(env.LOCAL,"linker_v8"); os.makedirs(OUT,exist_ok=True)
allv=np.ones(N,bool); COMP=components(allv)
TRG=train_groups7(); out=[]; H=int(sys.argv[1])
for f in ["F13","F47"]:
    g="F47" if f=="F13" else "F13"; sh=-2 if f=="F13" else 2
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=sorted(np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0],key=lambda i:NUM[i])
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}
        m=s+sh
        oth=set(np.where((FARM==g)&(NUM>=m-1)&(NUM<=m+10))[0].tolist())
        rem2=rem|oth
        touch=((FARM==f)&(NUM>=s-1)&(NUM<=s+10)); touch[list(oth)]=True
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False
        for r_ in blk:
            if not TRAIN[r_]: continue
            a0,p0=pick7(r_,H,lab,rem2,w)
            if a0 is None: continue
            out.append(dict(rec=rid(r_),prev=rid(a0),p=round(p0,4),prev_is_labelled=bool(lab[a0]),xf=FARM[a0]!=f,steps_to_anchor=1 if lab[a0] else -1,
                anchor_st23=round(ST[a0,23],2) if lab[a0] else np.nan,anchor_ec23=round(EC[a0,23],3) if lab[a0] else np.nan))
        print(f,s,flush=True)
O=pd.DataFrame(out); O.to_csv(os.path.join(OUT,"cv_all400_choice_v7m_h%d.csv"%H+""),index=False,encoding="utf-8-sig")

