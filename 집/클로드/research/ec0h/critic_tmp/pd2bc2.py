import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np
TR=pd.read_pickle(r"C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl")
TR["b_"]=TR.fb+"_"+TR.nb.astype(int).astype(str).str.zfill(3); TR["a_"]=TR.fa+"_"+TR.na.astype(int).astype(str).str.zfill(3)
T0=pd.read_pickle(r"C:\work\farmai\집\클로드\research\ec0h\critic_tmp\pd2b_T0.pkl")
tr=TR.drop_duplicates("b_").set_index("b_")
def tp(t): return (tr.at[t,"a_"], round(float(tr.at[t,"c1"]),2), bool(tr.at[t,"cert"])) if t in tr.index else None
T0["tp"]=[tp(t) for t in T0.t]
T0["ok"]=[ (x is not None and x[0]==p) for x,p in zip(T0.tp,T0.p)]
T0["known"]=[x is not None and x[2] for x in T0.tp]
print("truth cert n",T0.known.sum(),"of",len(T0),"| correct among cert",round(T0[T0.known].ok.mean(),2), "| correct any", round(T0.ok.mean(),2))
for t in ["F47_163","F47_108","F47_179","F13_233","F47_161","F13_153","F13_135","F13_137","F47_137","F13_062","F47_134","F13_052"]:
    print(t,"<-",T0.loc[t].p,"truth",tp(t), "x_sa %.3f r0 %.3f"%(T0.loc[t].x_sa,T0.loc[t].r0))
vc=T0.p.value_counts(); print("p shared:",vc[vc>1].to_dict())
p1=T0[T0.t.str[4:7].astype(int)<179]
print("|x_sa| pct F47_163 %.2f F47_108 %.2f"%((p1.x_sa.abs()<abs(T0.loc['F47_163'].x_sa)).mean(),(p1.x_sa.abs()<abs(T0.loc['F47_108'].x_sa)).mean()))
K=p1[p1.known]
for g,G in K.groupby("ok"): print("cert ok",g,"n",len(G),"slope r0~x_sa %.2f corr %.2f"%(np.polyfit(G.x_sa,G.r0,1)[0], np.corrcoef(G.x_sa,G.r0)[0,1]))
T0.to_pickle(r"C:\work\farmai\집\클로드\research\ec0h\critic_tmp\pd2b_T0b.pkl")
