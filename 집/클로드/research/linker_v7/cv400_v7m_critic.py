# MIRROR CV: hide chunk in ID f (s..s+9 + deleted neighbours) AND the other ID's mirror chunk (numbers shifted by -2 for F13->F47, +2 for F47->F13)
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v7"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
from feat7 import *
SC=r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\critic"
allv=np.ones(N,bool); COMP=components(allv)
TRG=train_groups7(); out=[]; frac=[]
for f in ["F13","F47"]:
    g="F47" if f=="F13" else "F13"; sh=-2 if f=="F13" else 2
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=sorted(np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0],key=lambda i:NUM[i])
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}
        m=s+sh
        oth=set(np.where((FARM==g)&(NUM>=m-1)&(NUM<=m+10))[0].tolist())
        dates={COMP[i] for i in blk}
        od=[i for i in np.where(FARM!=f)[0] if COMP[i] in dates]
        frac.append((len(od),sum(i in oth for i in od)))
        rem2=rem|oth
        touch=((FARM==f)&(NUM>=s-1)&(NUM<=s+10)); touch[list(oth)]=True
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False
        for r_ in blk:
            if not TRAIN[r_]: continue
            a0,p0=pick7(r_,0,lab,rem2,w)
            if a0 is None: continue
            out.append(dict(rec=rid(r_),prev=rid(a0),p=round(p0,4),prev_is_labelled=bool(lab[a0]),xf=FARM[a0]!=f,steps_to_anchor=1 if lab[a0] else -1,
                anchor_st23=round(ST[a0,23],2) if lab[a0] else np.nan,anchor_ec23=round(EC[a0,23],3) if lab[a0] else np.nan))
O=pd.DataFrame(out); O.to_csv(SC+r"\cv_all400_choice_v7m.csv",index=False,encoding="utf-8-sig")
fa=np.array(frac); print("other-ID same-date records of chunk: total %d, hidden by mirror %d (%.2f)"%(fa[:,0].sum(),fa[:,1].sum(),fa[:,1].sum()/fa[:,0].sum()))
c=O[(O.p>=.8)&O.prev_is_labelled]; print("mirror: 1-step labelled p>=.8: %d (cross-ID %d)"%(len(c),int(c.xf.sum())))
