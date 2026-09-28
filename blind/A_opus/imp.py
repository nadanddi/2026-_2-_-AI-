from cv import *
wcols=[c for c in F_ALL.columns if c.startswith("out_") or c.startswith("ot_") or c.startswith("orad")]
F=F_ALL[F_ALL.gh.isin(["F13","F47"])&F_ALL.sub_ec.notna()]
cols=feats(F,False,wcols)
m=lgb.train({**P,"num_leaves":7,"seed":0},lgb.Dataset(F[cols],F.sub_ec),1500)
imp=pd.Series(m.feature_importance("gain"),cols).sort_values(ascending=False); print((imp/imp.sum()).head(20).round(3))
F=F_ALL[F_ALL.gh.isin(["F13","F47"])&F_ALL.sub_temp.notna()]
cols=feats(F,False)
m=lgb.train({**P,"seed":0},lgb.Dataset(F[cols],F.sub_temp),800)
imp=pd.Series(m.feature_importance("gain"),cols).sort_values(ascending=False); print((imp/imp.sum()).head(15).round(3))
