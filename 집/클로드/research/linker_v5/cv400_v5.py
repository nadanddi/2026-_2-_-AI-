# linker v5 (2026-10-11 집 클로드): v4 WITHOUT sequential assignment - each evaluation record decides on its own (no 'claimed'
# blocking from earlier evaluation records' choices, no chains through evaluation-record choices).  Rule review: avoids using
# evaluation-derived decisions to update later decisions.  Only 1-step anchors (previous record labelled) are reported.
from feat4 import *
S5=r"C:\workarmai\집\클로드esearch\local\linker_v5"
import time; t0=time.time()
TRG=train_groups(); print("train groups",len(TRG),round(time.time()-t0),"s")
out=[]; bid=0
for f in ["F13","F47"]:
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=sorted(np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0],key=lambda i:NUM[i])
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}
        touch=(FARM==f)&(NUM>=s-1)&(NUM<=s+10)
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False
        claimed=set()   # v5: never updated
        for r_ in blk:
            a0,p0=pick(r_,0,lab,rem,claimed,w)
            if TRAIN[r_] and a0 is not None:
                st=1; x=a0; conf=p0; ok=bool(lab[x])   # v5: 1 step only
                out.append(dict(rec=rid(r_),prev=rid(a0),p=round(p0,4),prev_is_labelled=bool(lab[a0]),steps_to_anchor=st if ok else -1,anchor=rid(x) if ok else "",anchor_conf=round(conf,4),
                    anchor_st23=round(ST[x,23],2) if ok else np.nan,anchor_ec23=round(EC[x,23],3) if ok else np.nan,
                    dT_air=round(IN_[r_,0,0]-IN_[a0,23,0],2),fold_or_block_id=bid,block=f"{f}_{s}-{s+9}"))
        bid+=1
O=pd.DataFrame(out); O.to_csv(S5+r"\cv_all400_choice_v5.csv",index=False,encoding="utf-8-sig")
print("cv time",round(time.time()-t0),"s")
