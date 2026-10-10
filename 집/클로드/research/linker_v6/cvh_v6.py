# linker v6 hour-progressive CV: same blocks as cv400_v5; weights per decision hour; each record decides on its own.
from feat6 import *
HRS=(0,6,12,18,23); TRG={h:train_groups6(h) for h in HRS}; out=[]
for f in ["F13","F47"]:
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=sorted(np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0],key=lambda i:NUM[i])
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}
        touch=(FARM==f)&(NUM>=s-1)&(NUM<=s+10)
        lab=TRAIN.copy(); lab[touch]=False
        W={h:fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG[h] if not touch[b_] and not touch[a_]],l2=0.5,iters=300) for h in HRS}
        for r_ in blk:
            if not TRAIN[r_]: continue
            for h in HRS:
                a,p=pick6(r_,h,lab,rem,W[h])
                if a is None: continue
                out.append(dict(rec=rid(r_),h=h,prev=rid(a),p=round(p,4),prev_is_labelled=bool(lab[a]),anchor_ec23=round(EC[a,23],3) if lab[a] else np.nan,anchor_st23=round(ST[a,23],2) if lab[a] else np.nan,dT_air=round(IN_[r_,0,0]-IN_[a,23,0],2)))
O=pd.DataFrame(out); O.to_csv(S6+r"\cv_hourly_choice_v6.csv",index=False,encoding="utf-8-sig")
tm=dict(zip(TL.b,TL.a)); idx={rid(i):i for i in range(N)}
O["truth"]=[rid(tm[idx[r]]) if idx[r] in tm else "" for r in O.rec]
for h,g in O.groupby("h"):
    t=g[g.truth!=""]; c=g[(g.p>=.8)&g.prev_is_labelled]; tc=c[c.truth!=""]
    print("h%02d hit %.3f (n %d) | 1-step labelled p>=.8: %d records, hit %.3f (n %d)"%(h,(t.prev==t.truth).mean(),len(t),len(c),(tc.prev==tc.truth).mean() if len(tc) else np.nan,len(tc)))
