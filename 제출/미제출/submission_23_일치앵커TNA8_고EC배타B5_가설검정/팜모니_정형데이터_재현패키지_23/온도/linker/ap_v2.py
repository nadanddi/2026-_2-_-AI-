from common2 import *
w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG],l2=0.5,iters=300)
out=[]
for f in ["F13","F47"]:
    ev=sorted(np.where((FARM==f)&~TRAIN)[0],key=lambda i:NUM[i]); claimed=set(); chain={}
    for r_ in ev:
        a0,p0=pick(r_,0,TRAIN.copy(),set(),claimed,w)
        st=1; x=a0; conf=p0
        while not TRAIN[x] and x in chain: x,pp=chain[x]; conf*=pp; st+=1
        ok=bool(TRAIN[x])
        out.append(dict(row=rid(r_),prev_h0=rid(a0),p_h0=round(p0,3),prev_is_labelled=bool(TRAIN[a0]),steps_to_anchor=st if ok else -1,anchor=rid(x) if ok else "",anchor_conf=round(conf,3),
            anchor_ec23=round(EC[x,23],3) if ok else np.nan,anchor_st23=round(ST[x,23],2) if ok else np.nan,dT_air=round(IN_[r_,0,0]-IN_[a0,23,0],2)))
        a23,p23=pick(r_,23,TRAIN.copy(),set(),claimed,w)  # used only to block candidates for LATER records
        claimed.add(a23); chain[r_]=(a23,p23)
O=pd.DataFrame(out); O.to_csv(S+r"\eval60_predecessor_v2.csv",index=False,encoding="utf-8-sig")
C=O[(O.steps_to_anchor==1)&O.prev_is_labelled&(O.p_h0>=.8)]
C.to_csv(S+r"\eval60_confident_1step_anchor_v2.csv",index=False,encoding="utf-8-sig")
print(C[["row","prev_h0","p_h0","anchor_ec23","anchor_st23","dT_air"]].to_string()); print("n",len(C),"steps",O.steps_to_anchor.value_counts().sort_index().to_dict())
