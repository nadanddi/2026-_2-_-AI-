from feat4 import *
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
            a23,p23=pick(r_,23,lab,rem,claimed,w)
            if a23 is not None: claimed.add(a23); chain[r_]=(a23,p23)
        bid+=1
O=pd.DataFrame(out); O.to_csv(S4+r"\cv_all400_choice_v4.csv",index=False,encoding="utf-8-sig")
print("cv time",round(time.time()-t0),"s")
# --- compare with v3 (diagnosis uses full-data date components; diagnosis only)
allv=np.ones(N,bool); compA=components(allv)
idx={rid(i):i for i in range(N)}; tm=dict(zip(TL.b,TL.a))
def summ(path,nm):
    D=pd.read_csv(path,encoding="utf-8-sig"); res=[]
    for rr,pp,p in zip(D.rec,D.prev,D.p):
        b=idx[rr]
        if b not in tm: continue
        a=idx[pp]; t=tm[b]; seg=2 if NUM[b]>=179 else 1
        kind="hit" if a==t else ("date" if compA[a]!=compA[t] else "dong")
        res.append((seg,kind,p))
    R_=pd.DataFrame(res,columns=["seg","kind","p"])
    for sg in [1,2,0]:
        q=R_ if sg==0 else R_[R_.seg==sg]; c=q.kind.value_counts()
        hc=q[q.p>=.8]
        print(f"{nm} seg{'all' if sg==0 else sg}: n{len(q)} hit {c.get('hit',0)/len(q):.3f}  date-err {c.get('date',0)}  dong-err {c.get('dong',0)} | p>=.8 n{len(hc)} acc {(hc.kind=='hit').mean():.3f}")
    c1=D[(D.steps_to_anchor==1)&(D.p>=.8)]; print(f"{nm} 1step&p>=.8 rows {len(c1)}")
summ(S3+r"\cv_all400_choice_v3.csv","v3"); summ(S4+r"\cv_all400_choice_v4.csv","v4")
