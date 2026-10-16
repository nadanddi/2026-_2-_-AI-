"""DC4 training-only seasonal transform; independent PAV implementation."""
import numpy as np
import pandas as pd
W = ['out_temp', 'out_hum', 'out_rad', 'out_wspd']

def identify(raw):
    a = raw.copy()
    a['farm'] = a.row_id.str[:3]
    a['day'] = a.row_id.str[4:7].astype(int)
    a['hour'] = a.row_id.str[8:10].astype(int)
    return a

def pav(values):
    blocks = []
    for k, y in enumerate(values):
        blocks.append([k, k+1, float(y), 1])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
            b=blocks.pop();a=blocks.pop()
            blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    result=np.empty(len(values),float)
    for a,b,s,n in blocks:result[a:b]=s/n
    return result

def vectors(raw):
    answer={}
    for key,g in identify(raw).groupby(['farm','day'],sort=True):
        assert len(g)==24 and set(g.hour)==set(range(24))
        answer[key]=g.sort_values('hour')[W].to_numpy(float).T
    return answer

def mapping(tdays, qdays, vec):
    p1=[(f,int(d)) for f,d in tdays.itertuples(index=False,name=None) if d<179]
    reference=np.stack([vec[k] for k in p1])
    mu=np.nanmean(reference,axis=(0,2));sd=np.nanstd(reference,axis=(0,2));sd[sd==0]=1
    ref=(reference-mu[None,:,None])/sd[None,:,None]
    seasons={key:float(key[1]) for key in p1};fits={};notes={}
    for farm in ['F13','F47']:
        days=sorted(int(d) for f,d in tdays.itertuples(index=False,name=None) if f==farm and d>=179)
        x,y=[],[];nearest=[]
        for d in days:
            query=(vec[(farm,d)]-mu[:,None])/sd[:,None]
            distance=np.sqrt(np.nanmean((ref-query)**2,axis=(1,2)))
            nearest.append(float(p1[int(np.nanargmin(distance))][1]))
            selected=np.flatnonzero(distance<=.05)
            if len(selected):x.append(d);y.append(float(np.mean([p1[i][1] for i in selected])))
        count=len(x);fallback=count<2
        if fallback:x,y=days,nearest
        assert len(x)>0
        fitted=pav(y);x=np.asarray(x,float);fits[farm]=(x,fitted)
        for d in days:seasons[(farm,d)]=float(np.interp(d,x,fitted))
        notes[farm]={'exact_anchors':count,'training_late_days':len(days),'fallback':fallback,
                     'anchor_day':x.tolist(),'anchor_season_raw':list(y),'anchor_season_pav':fitted.tolist()}
    out=[]
    for farm,d in qdays.itertuples(index=False,name=None):
        if d<179:
            x=sorted(dd for f,dd in p1 if f==farm);out.append(float(np.interp(d,x,x)))
        else:out.append(float(np.interp(d,*fits[farm])))
    notes['standardization']={'weather_columns':W,'mu':mu.tolist(),'sd':sd.tolist(),
                              'source':'training first-period weather only'}
    return seasons,np.asarray(out),notes
