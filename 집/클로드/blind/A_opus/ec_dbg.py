from ec_struct import *
A,S=load(); folds=s2_folds(A)
H=A[A.gh.isin(["F13","F47"])]
ECd=H.groupby(["gh","day"]).sub_ec.mean(); ECh=H.set_index(["gh","day","hr"]).sub_ec
for g,B in folds:
    T=day_table(A,g).set_index("day")
    pr,info=predict_days(A,g,B,set(B),ECd,ECh,w=3,shape=False)
    for d in B:
        i=info[d]; tw=T.loc[d,"twin"]
        print(g,d,"cal",i["cal"],"twin",tw, "twinEC %.3f"%ECd.get((g,int(tw)),np.nan) if tw==tw else "", "lo %.3f hi %.3f pred %.3f true %.3f"%(i["lo"],i["hi"],i["lvl"],ECd[(g,d)]))
