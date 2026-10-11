# diagnostic: for pass-2 mirror-CV targets whose truth predecessor is a labelled pass-1 record, where does the truth rank and why.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v8")
from feat8 import *
import env
TLall=pd.read_pickle(S3+r"\truth.pkl"); tru={int(b):int(a) for a,b in zip(TLall.a1,TLall.b)}
TRG=train_groups7(); rows=[]
for f in ["F13","F47"]:
    g="F47" if f=="F13" else "F13"; sh=-2 if f=="F13" else 2
    lo,hi=NUM[FARM==f].min(),NUM[FARM==f].max()
    for s in range(lo,hi+1,10):
        blk=[i for i in np.where((FARM==f)&(NUM>=s)&(NUM<s+10))[0] if NUM[i]>=179 and TRAIN[i] and i in tru]
        if not blk: continue
        rem={i for i in np.where((FARM==f)&((NUM==s-1)|(NUM==s+10)))[0]}; m=s+sh
        oth=set(np.where((FARM==g)&(NUM>=m-1)&(NUM<=m+10))[0].tolist()); rem2=rem|oth
        touch=((FARM==f)&(NUM>=s-1)&(NUM<=s+10)); touch[list(oth)]=True
        w=fit_clogit([(X_,y_) for b_,a_,X_,y_ in TRG if not touch[b_] and not touch[a_]],l2=0.5,iters=300)
        lab=TRAIN.copy(); lab[touch]=False
        for b in blk:
            df,_=features7(b,0,lab,removed=rem2)
            if len(df)==0: continue
            sc=df[FEATS7].values.astype(float)@w; p=np.exp(sc-sc.max()); p/=p.sum(); df["p"]=p; df["c"]=[rid(int(a)) for a in df.a]
            t=tru[b]; tin=(df.a==t).any()
            o=df.sort_values("p",ascending=False).reset_index(drop=True)
            rk=int(np.where(o.a==t)[0][0])+1 if tin else -1
            top=o.iloc[0]; tr=o[o.a==t].iloc[0] if tin else None
            rows.append(dict(rec=rid(b),truth=rid(t),t_lab=bool(lab[t]),t_in=tin,rank=rk,p_truth=round(float(tr.p),3) if tin else np.nan,top=top.c,p_top=round(float(top.p),3),
                 **({("T_"+k):round(float(tr[k]),2) for k in ["oj","isprev","dsdev","d2c","d2w","excl","xf"]} if tin else {}),
                 **{("C_"+k):round(float(top[k]),2) for k in ["oj","isprev","dsdev","d2c","d2w","excl","xf"]}))
D=pd.DataFrame(rows); pd.set_option("display.width",250); print(D.to_string(index=False))
