# linker v6 (2026-10-11 집 클로드): v5 (no sequential assignment) + hour-progressive OPERATION FINGERPRINT feature.
# fp_h(r, a) = mean over hours 0..h and 8 indoor/actuator variables of |R[r,t,v] - R[a,t,v]| / sd_v, where R = residual of each
# variable regressed (per hour t) on the same hour's 4 outdoor variables; regressions and sd fitted on TRAINING records only.
# r uses only its own hours 0..h (causal); candidate a is an earlier record (full day allowed, same hours compared).
# Weights are trained separately for each decision hour h (MASK world, as v5).
from feat4 import *
import feat4 as F4
S6=S4.replace("linker_v4","linker_v6")
_R=np.full(VARS.shape,np.nan)
for t in range(24):
    Xo=np.c_[np.ones(N),X[:,t,:]]; ok_tr=TRAIN&np.isfinite(Xo).all(1)
    for v in range(VARS.shape[2]):
        y=VARS[:,t,v]; m=ok_tr&np.isfinite(y)
        if m.sum()<30: continue
        beta=np.linalg.lstsq(Xo[m],y[m],rcond=None)[0]
        fin=np.isfinite(Xo).all(1); _R[fin,t,v]=y[fin]-Xo[fin]@beta
_sdR=np.nanstd(_R[TRAIN].reshape(-1,VARS.shape[2]),0)+1e-6
def fp_dist(b,a,h):
    d=np.abs(_R[b,:h+1,:]-_R[a,:h+1,:])/_sdR
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
FEATS6=FEATS+["fp","fpna"]
def features6(b,h,labeled,claimed=set(),removed=set()):
    df,info=features(b,h,labeled,claimed=claimed,removed=removed)
    if len(df)==0: return df,info
    fp=np.array([fp_dist(b,a,h) for a in df.a.values])
    df["fpna"]=np.isnan(fp).astype(float); df["fp"]=np.nan_to_num(np.minimum(fp,6),nan=0.0)
    return df,info
def train_groups6(h):
    TRG=[]
    for b,a in zip(TL.b,TL.a):
        lab=TRAIN.copy(); lab[(FARM==FARM[b])&(np.abs(NUM-NUM[b])<=1)]=False
        df,_=features6(b,h,lab,removed=TEST_IDX)
        if len(df)==0: continue
        y=np.where(df.a.values==a)[0]
        if len(y): TRG.append((b,a,df[FEATS6].values.astype(float),y[0]))
    return TRG
def pick6(b,h,lab,removed,w):
    df,_=features6(b,h,lab,removed=removed)
    if len(df)==0: return None,0.0
    s=df[FEATS6].values.astype(float)@w; p=np.exp(s-s.max()); p/=p.sum(); k=int(np.argmax(p)); return int(df.a.values[k]),float(p[k])
