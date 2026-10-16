import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np, os
H=os.path.join(env.ROOT,u"집",u"클로드","research")
T=pd.read_pickle(os.path.join(env.LOCAL,'linker_v3','truth.pkl'))
rid={int(b):f"{f}_{n:03d}" for b,f,n in zip(T.b,T.fb,T.nb)}
print("truth rows",len(T),"farms",T.fb.value_counts().to_dict(), "max nb",T.nb.max())
tp={rid[int(b)]:(rid.get(int(a)) if a==a else None, bool(c)) for b,a,c in zip(T.b,T.a1,T.cert)}
R=pd.read_csv(os.path.join(H,"ev_validator","ec","out_v1","v2_ev_ec_rows.csv")); R=R[(R.arm=="BASE")&(R.seed.astype(str)!="avg")]
F=pd.read_csv(os.path.join(H,"ev_validator","ev_folds_v1.csv")); V=F[F.role=="val"].set_index("rec")
hid={k:set(G.rec) for k,G in F[F.role.isin(["val","del"])].groupby("fold")}
print(F.role.value_counts().to_dict())
Lm=pd.read_csv(os.path.join(env.LOCAL,"linker_v7","cv_all400_choice_v7m.csv")).set_index("rec")
anch=R.groupby("rec").anchored.first()
E=pd.read_csv(os.path.join(env.LOCAL,"linker_v7","eval60_predecessor_v7.csv")).set_index("row")
evalset=set(E.index)
rows=[]
for t in V.index:
    if t not in Lm.index or bool(anch.get(t,True)): continue
    pr=Lm.loc[t]
    if pr.prev_is_labelled or pr.xf or pr.p<.8: continue
    p=pr.prev
    if p not in V.index or V.at[p,"fold"]!=V.at[t,"fold"] or p[:3]!=t[:3] or int(p[4:7])>=int(t[4:7]): continue
    if bool(anch.get(p,False)): continue  # SA arm only
    k=V.at[t,"fold"]
    tt=tp.get(t,(None,False)); pt=tp.get(p,(None,False))
    a,ac=pt
    a_lab = a is not None and a not in hid[k] and a not in evalset
    pv=Lm.loc[p]
    rows.append(dict(t=t,p=p,fold=k,p2=int(t[4:7])>=179,t_link_true=tt[0]==p, t_cert=tt[1],a=a,a_cert=ac,a_lab=a_lab,
                     a_xfarm=(a is not None and a[:3]!=p[:3]), v7m_p_prev=pv.prev, v7m_p_ok=pv.prev==a, v7m_p_p=pv.p, chunk=V.at[t,"chunk"]))
D=pd.DataFrame(rows)
print("SA targets",len(D),"pass2",D.p2.sum())
for nm,G in D.groupby("p2"):
    print("pass2" if nm else "pass1", "n",len(G),"| t<-p truth-correct",int(G.t_link_true.sum()),
          "| p has truth prev a",int(G.a.notna().sum()),"cert",int((G.a.notna()&G.a_cert).sum()),
          "| a labelled in EV world",int(G.a_lab.sum()),"cert&lab",int((G.a_lab&G.a_cert).sum()),
          "| a other farm",int(G.a_xfarm.sum()),"| v7m(0h) picks a for p",int(G.v7m_p_ok.sum()),
          "| chunks",G.chunk.nunique(),"| chunks w/ cert&lab",G[G.a_lab&G.a_cert].chunk.nunique())
print(D[D.p2].to_string())
# true chain fully right & labelled
ok=D.t_link_true&D.a_lab&D.a_cert
print("pass1 decidable (t<-p true, a cert & labelled):",int((ok&~D.p2).sum()),"chunks",D[ok&~D.p2].chunk.nunique(),"; pass2:",int((ok&D.p2).sum()))
# eval records
print("\n--- eval60 ---")
cnt=[]
for r,pr in E.iterrows():
    conf1 = pr.prev_is_labelled and pr.p_h0>=.8
    sa = (not pr.prev_is_labelled) and (not pr.xf) and pr.p_h0>=.8 and pr.prev_h0[:3]==r[:3] and int(pr.prev_h0[4:7])<int(r[4:7])
    if sa:
        p=pr.prev_h0; pe=E.loc[p] if p in E.index else None
        cnt.append(dict(t=r,p=p,p_p=pr.p_h0,p_prev=None if pe is None else pe.prev_h0,p_prev_lab=None if pe is None else pe.prev_is_labelled,p_prev_p=None if pe is None else pe.p_h0,p_steps=None if pe is None else pe.steps_to_anchor))
C=pd.DataFrame(cnt); print("eval no-anchor SA-type targets:",len(C)); print(C.to_string())
print("\n--- information: SA estimate (EV EA2 p 23h) vs v3 estimate (a's true 23h EC, oracle-correct a) for t 0h ---")
piv={sd:G.pivot_table(index="rec",columns="hour",values="EA2") for sd,G in R.groupby("seed")}
Y=R.groupby(["rec","hour"]).y.first().unstack()
O=pd.read_csv(os.path.join(env.ROOT,u"연구실",u"클로드","results","ec_rv1_preds_v1.csv")); O=O[(O.tag=="CUR")&(O.validator=="DIAG10")]; O["rec"]=O.row_id.str[:7]
YD=O.groupby(["rec","hour"]).sub_ec.first().unstack()
for nm,G in (("all a_lab (any cert)",D[D.a_lab]),("cert&lab & t-link true",D[ok])):
    out=[]
    for _,r in G.iterrows():
        if r.a not in YD.index: continue
        sd=list(piv)[0]; P=piv[sd]
        out.append(dict(t=r.t,p2=r.p2,y0=Y.at[r.t,0],q0=P.at[r.t,0],sa=P.at[r.p,23],v3a=YD.at[r.a,23],yp23=Y.at[r.p,23]))
    Z=pd.DataFrame(out)
    rmse=lambda v: float(np.sqrt(np.mean(np.square(v))))
    print(nm,"n",len(Z),"| model q0",round(rmse(Z.y0-Z.q0),3),"| SA p23",round(rmse(Z.y0-Z.sa),3),"| a23 raw",round(rmse(Z.y0-Z.v3a),3),"| oracle y_p23",round(rmse(Z.y0-Z.yp23),3))
print("\n--- linker v6 hourly decision for p (all400 CV world): does it pick truth a, conf ---")
H6=pd.read_csv(os.path.join(env.LOCAL,'linker_v6','cv_hourly_choice_v6.csv'))
for h in (0,23):
    X=H6[H6.h==h].set_index("rec")
    for nm,G in (("a_lab",D[D.a_lab]),("cert&lab",D[D.a_lab&D.a_cert])):
        G=G[G.p.isin(X.index)]
        pick=X.loc[G.p,"prev"].values==G.a.values; conf=X.loc[G.p,"p"].values>=.8
        print(f"h{h} {nm}: n {len(G)} | picks a {pick.sum()} | p>=.8 {conf.sum()} | correct&conf {(pick&conf).sum()} | wrong&conf {(~pick&conf).sum()}")
# all-records: agreement of h0 vs h23 choices
X0=H6[H6.h==0].set_index("rec"); X23=H6[H6.h==23].set_index("rec"); com=X0.index.intersection(X23.index)
print("v6 h0 vs h23 same choice:", float((X0.loc[com,"prev"]==X23.loc[com,"prev"]).mean()), "n",len(com))
