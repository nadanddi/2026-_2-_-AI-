# linker v8 (2026-10-12 집 클로드): v7 + PASS-AWARE structural prior learned from TRAINING links only.
# Why: v7 weights are learned mostly from pass-1 links, where each ID interleaves two houses (same-ID number gap 2 dominates);
# in pass 2 the true previous date is a PASS-1 record in 73% of label-known links (41/56) and gap 1/2 only 27%, so v7 often picks
# the in-chunk gap-2 evaluation record (e.g. F47_209 -> F47_207 p .995, truth F47_161).  Fix: interaction features with the
# TARGET's pass (number >= 179, its own row id): t2*d1, t2*d2, t2*cand_in_pass1, t2*xf.  Same candidates, same visibility rules
# (same-ID earlier inputs + other-farm training only), same independent per-record decisions as v5/v7.
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v7"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
from feat7 import *
FEATS8=FEATS7+["t2d1","t2d2","t2c1","t2xf"]
def features8(b,h,labeled,removed=set()):
    D,info=features7(b,h,labeled,removed)
    if len(D)==0: return D,info
    t2=float(NUM[b]>=179)
    D["t2d1"]=t2*D.d1; D["t2d2"]=t2*(D.d2w+D.d2c); D["t2c1"]=t2*(NUM[D.a.values.astype(int)]<179).astype(float); D["t2xf"]=t2*D.xf
    return D,info
def train_groups8(exclude_touch=None):
    TRG=[]
    for b,a in zip(TL7.b,TL7.a):
        lab=TRAIN.copy(); lab[(FARM==FARM[b])&(np.abs(NUM-NUM[b])<=1)]=False
        df,_=features8(b,0,lab,removed=TEST_IDX)
        if len(df)==0: continue
        y=np.where(df.a.values==a)[0]
        if len(y): TRG.append((b,a,df[FEATS8].values.astype(float),y[0]))
    return TRG
def pick8(b,h,lab,removed,w):
    df,_=features8(b,h,lab,removed=removed)
    if len(df)==0: return None,0.0
    s=df[FEATS8].values.astype(float)@w; p=np.exp(s-s.max()); p/=p.sum(); k=int(np.argmax(p)); return int(df.a.values[k]),float(p[k])
