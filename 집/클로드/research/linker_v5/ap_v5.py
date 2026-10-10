# linker v5 (2026-10-11 집 클로드): v4 WITHOUT sequential assignment - each evaluation record decides on its own (no 'claimed'
# blocking from earlier evaluation records' choices, no chains through evaluation-record choices).  Rule review: avoids using
# evaluation-derived decisions to update later decisions.  Only 1-step anchors (previous record labelled) are reported.
import feat4 as F
from feat4 import *
S5=S4.replace("linker_v4","linker_v5")
esearch\local\linker_v5"
TRG=train_groups(); w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG],l2=0.5,iters=300)
np.save(S5+r"\w_v5.npy",w)
def run_farm(f):
    ev=sorted(np.where((FARM==f)&~TRAIN)[0],key=lambda i:NUM[i]); claimed=set(); out=[]   # v5: claimed never updated
    for r_ in ev:
        a0,p0=pick(r_,0,TRAIN.copy(),set(),claimed,w)
        st=1; x=a0; conf=p0; ok=bool(TRAIN[x])   # v5: 1 step only
        out.append(dict(row=rid(r_),prev_h0=rid(a0),p_h0=round(p0,3),prev_is_labelled=bool(TRAIN[a0]),steps_to_anchor=st if ok else -1,anchor=rid(x) if ok else "",anchor_conf=round(conf,3),
            anchor_ec23=round(EC[x,23],3) if ok else np.nan,anchor_st23=round(ST[x,23],2) if ok else np.nan,dT_air=round(IN_[r_,0,0]-IN_[a0,23,0],2)))
    return pd.DataFrame(out)
O=pd.concat([run_farm("F13"),run_farm("F47")],ignore_index=True)
O.to_csv(S5+r"\eval60_predecessor_v5.csv",index=False,encoding="utf-8-sig")
C=O[(O.steps_to_anchor==1)&O.prev_is_labelled&(O.p_h0>=.8)]
C.to_csv(S5+r"\eval60_confident_1step_anchor_v5.csv",index=False,encoding="utf-8-sig")
print(C[["row","prev_h0","p_h0","anchor_ec23","anchor_st23","dT_air"]].to_string()); print("n",len(C),"steps",O.steps_to_anchor.value_counts().sort_index().to_dict())
v4=pd.read_csv(S4+r"\eval60_predecessor_v4.csv",encoding="utf-8-sig")
m=O.merge(v4[["row","prev_h0","p_h0"]],on="row",suffixes=("","_v4")); print("changed vs v4:",(m.prev_h0!=m.prev_h0_v4).sum()); print(m[m.prev_h0!=m.prev_h0_v4][["row","prev_h0_v4","prev_h0","p_h0"]].to_string())
