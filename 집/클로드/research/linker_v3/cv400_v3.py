from common2 import *
out=[]; bid=0
for f in ["F13","F47"]:
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=sorted(np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0],key=lambda i:NUM[i])
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}
        touch=(FARM==f)&(NUM>=s-1)&(NUM<=s+10)
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False
        claimed=set(); chain={}
        for r_ in blk:
            a0,p0=pick(r_,0,lab,rem,claimed,w)
            if TRAIN[r_] and a0 is not None:
                st=1; x=a0; conf=p0
                while not lab[x] and x in chain: x,pp=chain[x]; conf*=pp; st+=1
                ok=bool(lab[x])
                out.append(dict(rec=rid(r_),prev=rid(a0),p=round(p0,4),prev_is_labelled=bool(lab[a0]),steps_to_anchor=st if ok else -1,anchor=rid(x) if ok else "",anchor_conf=round(conf,4),
                    anchor_st23=round(ST[x,23],2) if ok else np.nan,anchor_ec23=round(EC[x,23],3) if ok else np.nan,
                    dT_air=round(IN_[r_,0,0]-IN_[a0,23,0],2),fold_or_block_id=bid,block=f"{f}_{s}-{s+9}"))
            a23,p23=pick(r_,23,lab,rem,claimed,w)   # earlier block record's full day: usable by later records only
            if a23 is not None: claimed.add(a23); chain[r_]=(a23,p23)
        bid+=1
O=pd.DataFrame(out); O.to_csv(S+r"\cv_all400_choice_v3.csv",index=False,encoding="utf-8-sig")
print(len(O),O.prev_is_labelled.mean().round(3),O.steps_to_anchor.value_counts().sort_index().to_dict())
tm=dict(zip(TL.b,TL.a)); idx={rid(i):i for i in range(len(FARM))}
h=[tm[idx[r]]==idx[p] for r,p in zip(O.rec,O.prev) if idx[r] in tm]; print("T_LAB hit",np.mean(h).round(3),len(h))
c=O[(O.steps_to_anchor==1)&(O.p>=.8)]; print("1step p>=.8:",len(c))
