from feat2 import *
TL=pd.read_pickle(S+r"\truth.pkl"); TL=TL[TL.cert&(TL.fa==TL.fb)][["b","a1"]].rename(columns={"a1":"a"})
def score(df,w,claimed):
    s=df[FEATS].values.astype(float)@w; s=np.where(np.isin(df.a.values,list(claimed)),-np.inf,s)
    p=np.exp(s-s.max()); p/=p.sum(); return p
def pick(b,h,lab,removed,claimed,w):
    df,_=features(b,h,lab,removed=removed)
    if len(df)==0: return None,0.0
    p=score(df,w,claimed); k=int(np.argmax(p)); return int(df.a.values[k]),float(p[k])
TRG=[]
TEST_IDX=set(np.where(~TRAIN)[0].tolist())   # rule fix 10-11: weight training in the MASK world (evaluation records are not candidates)
for b,a in zip(TL.b,TL.a):
    lab=TRAIN.copy(); lab[(FARM==FARM[b])&(np.abs(NUM-NUM[b])<=1)]=False
    df,_=features(b,0,lab,removed=TEST_IDX)
    if len(df)==0: continue
    y=np.where(df.a.values==a)[0]
    if len(y): TRG.append((b,a,df[FEATS].values.astype(float),y[0]))
rid=lambda i:f"{FARM[i]}_{NUM[i]:03d}"
