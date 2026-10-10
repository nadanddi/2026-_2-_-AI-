# linker v7 PESSIMISTIC CV (test-like geometry across BOTH IDs): besides the 10-record chunk and its two same-farm neighbours,
# also hide (inputs and labels) every other-farm record whose date equals the date of any hidden-chunk record or of its two
# deleted neighbours.  Mimics the evaluation design where the other ID's same-period records are evaluation records.
from feat7 import *
allv=np.ones(N,bool); COMP=components(allv)   # validation-construction only (dates of training records)
TRG=train_groups7(); out=[]
for f in ["F13","F47"]:
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=sorted(np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0],key=lambda i:NUM[i])
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}
        dates={COMP[i] for i in list(blk)+list(rem)}
        other=[i for i in np.where(FARM!=f)[0] if COMP[i] in dates]
        rem2=rem|set(other)
        touch=((FARM==f)&(NUM>=s-1)&(NUM<=s+10)); touch[other]=True
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False
        for r_ in blk:
            if not TRAIN[r_]: continue
            a0,p0=pick7(r_,0,lab,rem2,w)
            if a0 is None: continue
            out.append(dict(rec=rid(r_),prev=rid(a0),p=round(p0,4),prev_is_labelled=bool(lab[a0]),xf=FARM[a0]!=f,steps_to_anchor=1 if lab[a0] else -1,
                anchor_st23=round(ST[a0,23],2) if lab[a0] else np.nan,anchor_ec23=round(EC[a0,23],3) if lab[a0] else np.nan,dT_air=round(IN_[r_,0,0]-IN_[a0,23,0],2)))
O=pd.DataFrame(out); O.to_csv(S7+r"\cv_all400_choice_v7p.csv",index=False,encoding="utf-8-sig")
c=O[(O.p>=.8)&O.prev_is_labelled]; print("pessimistic: 1-step labelled p>=.8: %d (cross-ID %d)"%(len(c),int(c.xf.sum())))
