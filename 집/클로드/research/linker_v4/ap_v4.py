import feat4 as F
from feat4 import *
TRG=train_groups(); w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG],l2=0.5,iters=300)
np.save(S4+r"\w_v4.npy",w)
def run_farm(f):
    ev=sorted(np.where((FARM==f)&~TRAIN)[0],key=lambda i:NUM[i]); claimed=set(); chain={}; out=[]
    for r_ in ev:
        a0,p0=pick(r_,0,TRAIN.copy(),set(),claimed,w)
        st=1; x=a0; conf=p0
        while not TRAIN[x] and x in chain: x,pp=chain[x]; conf*=pp; st+=1
        ok=bool(TRAIN[x])
        out.append(dict(row=rid(r_),prev_h0=rid(a0),p_h0=round(p0,3),prev_is_labelled=bool(TRAIN[a0]),steps_to_anchor=st if ok else -1,anchor=rid(x) if ok else "",anchor_conf=round(conf,3),
            anchor_ec23=round(EC[x,23],3) if ok else np.nan,anchor_st23=round(ST[x,23],2) if ok else np.nan,dT_air=round(IN_[r_,0,0]-IN_[a0,23,0],2)))
        a23,p23=pick(r_,23,TRAIN.copy(),set(),claimed,w)   # blocks candidates for LATER records only
        claimed.add(a23); chain[r_]=(a23,p23)
    return pd.DataFrame(out)
O=pd.concat([run_farm("F13"),run_farm("F47")],ignore_index=True)
O.to_csv(S4+r"\eval60_predecessor_v4.csv",index=False,encoding="utf-8-sig")
C=O[(O.steps_to_anchor==1)&O.prev_is_labelled&(O.p_h0>=.8)]
C.to_csv(S4+r"\eval60_confident_1step_anchor_v4.csv",index=False,encoding="utf-8-sig")
print(C[["row","prev_h0","p_h0","anchor_ec23","anchor_st23","dT_air"]].to_string()); print("n",len(C),"steps",O.steps_to_anchor.value_counts().sort_index().to_dict())
v2=pd.read_csv(r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\96fe1d15-2691-4cbd-a117-4a3e2d96bdf9\scratchpad\linker\eval60_predecessor_v2.csv",encoding="utf-8-sig")
m=O.merge(v2[["row","prev_h0"]],on="row",suffixes=("","_v2")); print("changed vs v2:",(m.prev_h0!=m.prev_h0_v2).sum(), m[m.prev_h0!=m.prev_h0_v2][["row","prev_h0_v2","prev_h0","p_h0"]].to_string())
# ---- isolation tests
rng=np.random.default_rng(0)
def perturb(mask):
    for A in (F.X,F.IN_,F.AC):
        A[mask]=A[mask]+rng.normal(0,3,A[mask].shape)
    F.refresh(); F._cal_cache.clear()
base={f:run_farm(f) for f in ["F13","F47"]}
saveX,saveI,saveA=F.X.copy(),F.IN_.copy(),F.AC.copy()
def restore():
    F.X[:]=saveX; F.IN_[:]=saveI; F.AC[:]=saveA; F.refresh(); F._cal_cache.clear()
for f,o in [("F13","F47"),("F47","F13")]:
    perturb((FARM==o)&~TRAIN); new=run_farm(f); restore()
    print(f"test1 {o} eval inputs perturbed -> {f} picks changed: {(new.prev_h0!=base[f].prev_h0).sum()}, p changed: {(np.abs(new.p_h0-base[f].p_h0)>1e-9).sum()}")
for f,cut in [("F13",220),("F47",218)]:
    perturb((FARM==f)&~TRAIN&(NUM>=cut)); new=run_farm(f); restore()
    e=new[[int(x[4:]) <cut for x in new.row]]; b0=base[f][[int(x[4:])<cut for x in base[f].row]]
    print(f"test2 {f} eval >= {cut} perturbed -> earlier picks changed: {(e.prev_h0.values!=b0.prev_h0.values).sum()} of {len(e)}, p changed: {(np.abs(e.p_h0.values-b0.p_h0.values)>1e-9).sum()}")
