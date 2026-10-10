# linker v7 CV: same blocks as cv400_v5 (10-record chunk hidden, both same-farm neighbours deleted), independent decisions.
from feat7 import *
import time; t0=time.time()
TRG=train_groups7(); print("train groups",len(TRG),"cross-ID truths",int(sum(FARM[b]!=FARM[a] for b,a,_,_ in TRG)))
out=[]
for f in ["F13","F47"]:
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=sorted(np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0],key=lambda i:NUM[i])
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}
        touch=(FARM==f)&(NUM>=s-1)&(NUM<=s+10)
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False
        for r_ in blk:
            if not TRAIN[r_]: continue
            a0,p0=pick7(r_,0,lab,rem,w)
            if a0 is None: continue
            out.append(dict(rec=rid(r_),prev=rid(a0),p=round(p0,4),prev_is_labelled=bool(lab[a0]),xf=FARM[a0]!=f,steps_to_anchor=1 if lab[a0] else -1,
                anchor_st23=round(ST[a0,23],2) if lab[a0] else np.nan,anchor_ec23=round(EC[a0,23],3) if lab[a0] else np.nan,dT_air=round(IN_[r_,0,0]-IN_[a0,23,0],2)))
O=pd.DataFrame(out); O.to_csv(S7+r"\cv_all400_choice_v7.csv",index=False,encoding="utf-8-sig")
tm=dict(zip(TL7.b,TL7.a)); idx={rid(i):i for i in range(N)}
O["truth"]=[rid(tm[idx[r]]) if idx[r] in tm else "" for r in O.rec]; T=O[O.truth!=""]
T2=T[T.rec.str[4:7].astype(int)>=179]; c=O[(O.p>=.8)&O.prev_is_labelled]
print("cv time",round(time.time()-t0),"s | hit all %.3f (n %d), pass-2 %.3f (n %d), cross-ID truths hit %.3f (n %d) | 1-step labelled p>=.8: %d (cross-ID %d), acc %.3f"%(
    (T.prev==T.truth).mean(),len(T),(T2.prev==T2.truth).mean(),len(T2),(T[T.truth.str[:3]!=T.rec.str[:3]].prev==T[T.truth.str[:3]!=T.rec.str[:3]].truth).mean(),(T.truth.str[:3]!=T.rec.str[:3]).sum(),len(c),int(c.xf.sum()),(c[c.truth!=""].prev==c[c.truth!=""].truth).mean()))
