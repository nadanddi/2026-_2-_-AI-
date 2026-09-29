from cv import *
import sys
yy=[];pp=[]
for g,B in FOLDS:
    v=F_ALL[(F_ALL.gh==g)&F_ALL.day.isin(B)&F_ALL.sub_temp.notna()]
    yy.append(v.sub_temp.values); pp.append(v.in_temp.values)
print("T0 sub_temp=in_temp pooled", np.sqrt(np.mean((np.concatenate(yy)-np.concatenate(pp))**2)))
print("folds:", [f"{g}{B[0]}" for g,B in FOLDS])
M,o=run("sub_temp",twin=False); summarize("T1 F13F47 no-twin",M,o)
M,o=run("sub_temp",twin=True); summarize("T2 F13F47 twin",M,o)
M,o=run("sub_temp",twin=True,resid="in_temp"); summarize("T2r resid in_temp",M,o)
