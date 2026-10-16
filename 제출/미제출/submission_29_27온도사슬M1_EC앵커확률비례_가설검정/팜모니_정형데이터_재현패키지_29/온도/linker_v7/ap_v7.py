# package copy of research linker_v7/ap_v7.py
# linker v7 evaluation decisions: each evaluation record independently at hour 0 (no claimed, no chains).
from feat7 import *
TRG=train_groups7(); w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG],l2=0.5,iters=300); np.save(S7+r"\w_v7.npy",w)
out=[]
for f in ["F13","F47"]:
    for r_ in sorted(np.where((FARM==f)&~TRAIN)[0],key=lambda i:NUM[i]):
        a0,p0=pick7(r_,0,TRAIN.copy(),set(),w)
        out.append(dict(row=rid(r_),prev_h0=rid(a0),p_h0=round(p0,3),prev_is_labelled=bool(TRAIN[a0]),xf=FARM[a0]!=f,steps_to_anchor=1 if TRAIN[a0] else -1,
            anchor_ec23=round(EC[a0,23],3) if TRAIN[a0] else np.nan,anchor_st23=round(ST[a0,23],2) if TRAIN[a0] else np.nan,dT_air=round(IN_[r_,0,0]-IN_[a0,23,0],2)))
O=pd.DataFrame(out); O.to_csv(S7+r"\eval60_predecessor_v7.csv",index=False,encoding="utf-8-sig")
C=O[(O.steps_to_anchor==1)&O.prev_is_labelled&(O.p_h0>=.8)]; C.to_csv(S7+r"\eval60_confident_1step_anchor_v7.csv",index=False,encoding="utf-8-sig")
print(C.to_string()); print("n",len(C),"cross-ID",int(C.xf.sum()))
