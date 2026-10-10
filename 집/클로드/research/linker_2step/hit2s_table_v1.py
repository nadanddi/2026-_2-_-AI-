# hit-rate table for 2-step mirror CV choices (reads cv_mirror_choice_2step_v1.csv)
import sys; sys.path.insert(0, "C:/work/farmai/집/클로드/research"); import env
import pandas as pd, numpy as np
S2S=r"C:\work\farmai\집\클로드\research\local\linker_2step"
O=pd.read_csv(S2S+r"\cv_mirror_choice_2step_v1.csv")
C1=pd.read_csv(r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\critic\cv_all400_choice_v7m.csv")
one=set(C1[(C1.p>=.8)&C1.prev_is_labelled.astype(bool)].rec); O["one"]=O.rec.isin(one)
O["bin"]=pd.cut(O.p,[0,.5,.8,.9,1.01],right=False)
T=O[O.truth_a2.notna()]
print("records",len(O),"with pick",O.pick.notna().sum(),"truth-2step",len(T),"truth a2 available in fold",int(T.a2_available.sum()))
for nm,Z in [("all truth",T),("truth & a2 available",T[T.a2_available==True]),("truth & a2 avail & a1 hidden",T[(T.a2_available==True)&(T.a1_labelled_in_fold==False)]),
             ("truth & a2 avail & no 1-step anchor",T[(T.a2_available==True)&~T.one])]:
    g=Z.groupby("bin",observed=False).agg(n=("rec","size"),hit=("hit",lambda s:np.nansum(s.astype(float))))
    print("--",nm,"n",len(Z),"hits",int(np.nansum(Z.hit.astype(float))),"\n",g.assign(rate=(g.hit/g.n).round(3)).to_string())
P=O[(O.p>=.8)&~O.one]; print("-- p>=.8 & no 1-step anchor: n",len(P),"with truth",P.truth_a2.notna().sum(),"hits",int(np.nansum(P.hit.astype(float))),"xf",int(P.xf.sum()),"pass2",int((P.rec.str[4:].astype(int)>=179).sum()))
print(P[["rec","pick","p","xf","truth_a2","hit","a2_available","a1_labelled_in_fold"]].to_string(index=False))
