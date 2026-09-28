"""Re-validation of the final configuration with training-row features built from test-masked inputs (as in final.py)."""
from features import *
from final import P, DROP, TEMP_CFG, EC_CFG, masked_inputs
import lightgbm as lgb, pickle
A,S=load(); FOLDS=s2_folds(A)
F=build(A,["F13","F47"]); Am=masked_inputs(A); Fm=build(Am,["F13","F47"])
wcols=[c for c in F.columns if c.startswith("out_") or c.startswith("ot_") or c.startswith("orad")]
SEEDS=(0,1,2)
rows=[]; oof={}
for g,B in FOLDS:
    hidden={(g,d) for d in B}
    Fv=add_twin(F,A,hidden); Ft=add_twin(Fm,Am,hidden)
    va=Fv[(Fv.gh==g)&Fv.day.isin(B)]
    tr=Ft[~((Ft.gh==g)&Ft.day.isin(B))&(Ft.is_test==0)]
    cols=[c for c in Fv.columns if c not in DROP]
    res={}
    d=tr[tr.sub_temp.notna()]
    for s in SEEDS:
        ps=[]
        for c in TEMP_CFG:
            m=lgb.train({**P,**c["p"],"seed":s},lgb.Dataset(d[cols],d.sub_temp),c["n"]); ps.append(m.predict(va[cols]))
        res[("temp",s)]=np.mean(ps,0)
    d=tr[tr.sub_ec.notna()]
    for s in SEEDS:
        ps=[]
        for c in EC_CFG:
            cc=[x for x in cols if x not in wcols and (c["twin"] or not x.startswith("twin"))]
            m=lgb.train({**P,**c["p"],"seed":s},lgb.Dataset(d[cc],d.sub_ec),c["n"]); ps.append(m.predict(va[cc]))
        res[("ec",s)]=np.mean(ps,0)
    # baselines
    res[("temp","in_temp")]=va.in_temp.values
    res[("ec","mean")]=np.full(len(va),d.sub_ec.mean())
    res[("ec","gh_mean")]=np.full(len(va),d[d.gh==g].sub_ec.mean())
    oof[(g,B[0])]=(va.sub_temp.values,va.sub_ec.values,res)
    print(g,B[0],"temp",[round(np.sqrt(np.mean((res[("temp",s)]-va.sub_temp.values)**2)),3) for s in SEEDS],
          "ec",[round(np.sqrt(np.mean((res[("ec",s)]-va.sub_ec.values)**2)),3) for s in SEEDS],flush=True)
pickle.dump(oof,open("res_final_cv.pkl","wb"))
def pooled(t,k):
    i=0 if t=="temp" else 1
    y=np.concatenate([v[i] for v in oof.values()]); p=np.concatenate([v[2][(t,k)] for v in oof.values()])
    return np.sqrt(np.mean((y-p)**2))
for s in SEEDS: print("seed",s,"temp %.4f ec %.4f"%(pooled("temp",s),pooled("ec",s)))
print("baselines: temp=in_temp %.4f | ec global mean %.4f | ec gh mean %.4f"%(pooled("temp","in_temp"),pooled("ec","mean"),pooled("ec","gh_mean")))
