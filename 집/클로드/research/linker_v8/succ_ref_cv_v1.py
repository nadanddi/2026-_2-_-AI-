# SR1 (2026-10-12 집 클로드; user: "허용 가정하고 뒤쪽 학습 기록 참조 시험"): SUCCESSOR anchor from LATER TRAINING records,
# ASSUMING organisers allow referencing later-dated training records (train_X/train_y).  Never used: other evaluation records'
# inputs, the target's inputs after hour h.  Mirror-CV geometry identical to linker v7 (chunk + deleted neighbours + mirror chunk
# hidden; their labels hidden, and their inputs not used except the target's own).
# Rule fixed before running:
#  date D of target t at hour h: labelled-visible records j with all 4 hour-0 outdoor cells equal to t's (same date);
#     if none -> no successor (unless h == 23: use t's own 23 h outdoor).
#  date D+1: component (full-day outdoor >= 90% equal) of LABELLED records minimising midnight cost from j's (or t's) 23 h outdoor
#     to member 0 h outdoor (v7 oj formula); require cost < 1.0.
#  candidates s = labelled members of D+1 whose predecessor is NOT already label-known (label_successor(lab) values excluded).
#  choice: h == 23 -> indoor/actuator midnight continuity t(23)->s(0) (v7 dv sum);  h < 23 -> operation fingerprint fp_dist(t, s, h)
#     (linker v6 residual fingerprint, training-fitted).  Accept if 1 candidate, or best score margin >= M (M = 0.3 fingerprint,
#     1.0 continuity).
#  anchor (pre-fixed from NX2 6.591 / 6.602 slopes): EC pred_h = p_h + 0.8 (S0_ec - p_h) for all hours from h on;
#     temperature pred_h = q_h + b_h (S0_st - q_h), b_21 .1, b_22 .2, b_23 .6, else 0.  Applied only where the existing previous-date
#     anchor rule is NOT active (prev not confident-labelled).
# Reported: acceptance count, accuracy vs certain truth successor, EC/temp RMSE change on affected rows (pass 1 / pass 2), records.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v8"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6")
from feat8 import *
import feat6 as F6
import env, importlib.util
Xf=X.reshape(N,-1)
compF=np.arange(N)
par=list(range(N))
def fd(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for i in range(N):
    eq=(np.abs(Xf[i+1:]-Xf[i])<1e-6).mean(1)
    for j in np.where(eq>=.9)[0]+i+1: par[fd(j)]=fd(i)
compF=np.array([fd(i) for i in range(N)])
TLa=pd.read_pickle(S3+r"\truth.pkl"); succ_true={}
for a,b,c in zip(TLa.a1,TLa.b,TLa.cert):
    if c: succ_true[int(a)]=int(b)
def mcost(a23,b0): return float(np.sum(np.array([1,1,.3,.5])*np.abs(a23-b0)/sdO))
TRG=train_groups7(); rows=[]
for f in ["F13","F47"]:
    g="F47" if f=="F13" else "F13"; sh=-2 if f=="F13" else 2
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s0 in range(lo,hi+1,10):
        blk=[i for i in np.where((FARM==f)&(NUM>=s0)&(NUM<s0+10))[0] if TRAIN[i]]
        if not blk: continue
        m=s0+sh; oth=set(np.where((FARM==g)&(NUM>=m-1)&(NUM<=m+10))[0].tolist())
        touch=((FARM==f)&(NUM>=s0-1)&(NUM<=s0+10)); touch[list(oth)]=True
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False; known_pred=set(succ_for(lab).values())
        labidx=np.where(lab)[0]
        for t in blk:
            a0,p0=pick7(t,0,lab,(touch.nonzero()[0].tolist() and set(np.where(touch)[0])-{t}) or set(),w)
            prev_conf=bool(a0 is not None and p0>=.8 and lab[a0])
            for h in (0,6,12,18,23):
                same=[j for j in labidx if (np.abs(X[j,0]-X[t,0])<1e-6).all()]
                if same: a23=X[same[0],23]
                elif h==23: a23=X[t,23]
                else: rows.append(dict(t=rid(t),h=h,status="nodate",prev_conf=prev_conf)); continue
                comps={}
                for j in labidx: comps.setdefault(compF[j],[]).append(j)
                best=min(comps.items(),key=lambda kv:mcost(a23,X[kv[1][0],0]))
                if mcost(a23,X[best[1][0],0])>=1.0 or compF[t]==best[0]: rows.append(dict(t=rid(t),h=h,status="nonext",prev_conf=prev_conf)); continue
                cand=[s for s in best[1] if s not in known_pred]
                if not cand: rows.append(dict(t=rid(t),h=h,status="allexcl",prev_conf=prev_conf)); continue
                if h==23: sc=np.array([np.nansum(np.minimum(np.abs(VARS[s,0]-VARS[t,23])/sdV,6)) for s in cand]); M=1.0
                else: sc=np.array([F6.fp_dist(t,s,h) for s in cand]); sc=np.nan_to_num(sc,nan=9); M=0.3
                o=np.argsort(sc); k=cand[o[0]]; marg=(sc[o[1]]-sc[o[0]]) if len(cand)>1 else 9
                acc=marg>=M
                tru=succ_true.get(t)
                rows.append(dict(t=rid(t),h=h,status="acc" if acc else "lowmargin",prev_conf=prev_conf,succ=rid(k),ncand=len(cand),
                     correct=(None if tru is None else (k==tru)),S0_ec=EC[k,0],S0_st=ST[k,0]))
        print(f,s0,flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(env.LOCAL,"linker_v8","sr1_succ_ref_cv.csv"),index=False)
print(D.groupby(["h","status"]).size().unstack(fill_value=0))
A=D[D.status=="acc"]; print("accepted, accuracy vs certain truth:",A.groupby("h").correct.apply(lambda s:"%.2f (n %d)"%(s.dropna().astype(float).mean(),s.notna().sum())).to_dict())
