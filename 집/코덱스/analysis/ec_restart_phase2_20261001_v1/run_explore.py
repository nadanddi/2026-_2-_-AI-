from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import csv, hashlib, json, math, platform
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import linalg

OUT=Path(__file__).resolve().parent
DATA=Path(env.DATA)
LOCK=Path(env.CODEX)/'ec_final_lock'/'locked_days.json'
RAW=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
SENS=['in_temp','in_hum','in_co2']
ACT=RAW[7:]
SEED=261001

def read():
    lock={(z['farm'],int(z['day'])) for z in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    x=pd.read_csv(DATA/'train_X.csv',usecols=['row_id']+RAW)
    k=x.row_id.str.split('_',expand=True)
    x=x.assign(farm=k[0],day=k[1].astype(int),hour=k[2].astype(int))
    x=x[x.farm.isin(['F13','F47'])].copy()
    x=x[[ (f,int(d)) not in lock for f,d in zip(x.farm,x.day) ]].copy()
    rows=[]
    with (DATA/'train_y.csv').open(newline='',encoding='utf-8-sig') as file:
        for r in csv.DictReader(file):
            f,d,h=r['row_id'].split('_')
            if f not in ['F13','F47'] or (f,int(d)) in lock: continue
            if r['sub_ec'].strip(): rows.append((r['row_id'],float(r['sub_ec'])))
    y=pd.DataFrame(rows,columns=['row_id','sub_ec'])
    assert x.row_id.is_unique and y.row_id.is_unique
    d=x.merge(y,on='row_id',validate='one_to_one')
    assert len(d)==8640 and (d.groupby(['farm','day']).size()==24).all()
    return x,d

def features(x,h):
    out=[]
    for (f,d),g in x.groupby(['farm','day'],sort=True):
        g=g[g.hour.le(h)].sort_values('hour')
        assert len(g)==h+1
        z={'farm':f,'day':int(d)}
        for c in RAW: z[c+'_current']=float(g.iloc[-1][c])
        z['sensor_missing_count']=int(g[SENS].isna().sum().sum())
        if h==6:
            for c in ACT: z[c+'_mean']=float(g[c].mean())
            for c in SENS:
                z[c+'_rough']=float(g[c].diff().abs().mean())
                z[c+'_delta']=float(g.iloc[-1][c]-g.iloc[0][c])
        out.append(z)
    return pd.DataFrame(out)

def controls(d,initial,increment):
    f=d.farm.eq('F47').to_numpy(float); day=d.day.to_numpy(float)/100; late=d.day.ge(179).to_numpy(float)
    a=np.column_stack([np.ones(len(d)),f,day,f*day,late,f*late])
    if increment:
        b=initial[[c+'_current' for c in RAW]].to_numpy(float)
        missing=~np.isfinite(b)
        means=np.nanmean(b,axis=0)
        b=np.where(missing,means,b)
        sd=b.std(axis=0); sd[sd<1e-12]=1
        b=(b-b.mean(axis=0))/sd
        a=np.column_stack([a,b,missing.astype(float)])
    return a

def residual_pair(xx,yy,a,method='numpy'):
    valid=np.isfinite(xx)&np.isfinite(yy)&np.isfinite(a).all(axis=1)
    aa=a[valid]; xx=xx[valid]; yy=yy[valid]
    if len(xx)<15: return valid,None,None,None
    if method=='numpy':
        bx=np.linalg.lstsq(aa,xx,rcond=None)[0]; by=np.linalg.lstsq(aa,yy,rcond=None)[0]
    else:
        bx=linalg.lstsq(aa,xx,cond=None,lapack_driver='gelsd')[0]; by=linalg.lstsq(aa,yy,cond=None,lapack_driver='gelsd')[0]
    rx=xx-aa@bx; ry=yy-aa@by
    if np.linalg.norm(rx)<1e-9 or np.linalg.norm(ry)<1e-9: return valid,None,None,None
    r=float(np.dot(rx,ry)/np.sqrt(np.dot(rx,rx)*np.dot(ry,ry)))
    return valid,rx,ry,r

def stratum_r(frame,initial,xx,yy,inc,mask):
    pos=np.flatnonzero(mask)
    a=controls(frame.iloc[pos],initial.iloc[pos],inc)
    return residual_pair(xx[pos],yy[pos],a)[3]

def verify_features(x,fs):
    checks=[]
    for h,expected in fs.items():
        base=expected.sort_values(['farm','day']).reset_index(drop=True)
        changed=x.copy(); late=changed.hour.gt(h)
        changed.loc[late,RAW]=changed.loc[late,RAW]*13+777
        pd.testing.assert_frame_equal(base,features(changed,h))
        pd.testing.assert_frame_equal(base,features(x[x.hour.le(h)],h))
        pd.testing.assert_frame_equal(base,features(x.sample(frac=1,random_state=SEED),h))
        other=x.copy(); other.loc[other.farm.eq('F47'),RAW]=999
        pd.testing.assert_frame_equal(base[base.farm.eq('F13')].reset_index(drop=True),features(other,h).query("farm=='F13'").reset_index(drop=True))
        checks.extend([f'h{h}_future_corrupt',f'h{h}_future_delete',f'h{h}_row_shuffle',f'h{h}_other_farm'])
    # csv parsing + math independent feature reconstruction for all days.
    buckets=defaultdict(dict)
    ids=set(x.row_id)
    with (DATA/'train_X.csv').open(newline='',encoding='utf-8-sig') as file:
        for r in csv.DictReader(file):
            if r['row_id'] not in ids: continue
            f,d,h=r['row_id'].split('_')
            if int(h)<=6: buckets[(f,int(d))][int(h)]={c:float(r[c]) if r[c] else None for c in RAW}
    for h,expected in fs.items():
        for _,r in expected.iterrows():
            b=buckets[(r.farm,int(r.day))]
            z={c+'_current':b[h][c] for c in RAW}
            z['sensor_missing_count']=sum(b[t][c] is None for t in range(h+1) for c in SENS)
            if h==6:
                for c in ACT:
                    v=[b[t][c] for t in range(7) if b[t][c] is not None]
                    z[c+'_mean']=math.fsum(v)/len(v) if v else None
                for c in SENS:
                    v=[abs(b[t][c]-b[t-1][c]) for t in range(1,7) if b[t][c] is not None and b[t-1][c] is not None]
                    z[c+'_rough']=math.fsum(v)/len(v) if v else None
                    z[c+'_delta']=b[6][c]-b[0][c] if b[6][c] is not None and b[0][c] is not None else None
            for c,v in z.items():
                if v is None: assert pd.isna(r[c])
                else: assert math.isclose(v,float(r[c]),rel_tol=1e-11,abs_tol=1e-9),(h,c)
    checks.append('all_features_csv_math')
    return checks

def analyze(x,d,verify=False):
    fs={h:features(x,h) for h in [0,6]}
    frame=d.groupby(['farm','day'],sort=True).sub_ec.mean().reset_index()
    initial=fs[0]; assert frame[['farm','day']].equals(initial[['farm','day']])
    yy=frame.sub_ec.to_numpy(float)
    cluster=pd.factorize(frame.farm+'_'+(frame.day//5).astype(str),sort=True)[0]
    nc=int(cluster.max()+1)
    top5=np.argsort(yy,kind='stable')[-5:]
    rows=[]; arrays=[]; scores=[]; ver=[]
    for h,inc in [(0,False),(6,False),(6,True)]:
        z=fs[h]; a=controls(frame,initial,inc)
        for c in z.columns[2:]:
            xx=z[c].to_numpy(float)
            valid,rx,ry,r=residual_pair(xx,yy,a)
            raw=float(np.corrcoef(xx[valid],yy[valid])[0,1]) if xx[valid].std()>1e-12 else None
            row={'hour':h,'incremental':inc,'feature':c,'n':int(valid.sum()),'r_raw':raw,'r_partial':r}
            for name,m in [('F13',frame.farm.eq('F13').to_numpy()),('F47',frame.farm.eq('F47').to_numpy()),
                           ('early',frame.day.lt(179).to_numpy()),('late',frame.day.ge(179).to_numpy()),
                           ('without_high_ec',yy<1.2),('without_top5',~np.isin(np.arange(len(yy)),top5))]:
                row['r_'+name]=stratum_r(frame,initial,xx,yy,inc,m)
            xxall=np.zeros(len(yy)); yyall=np.zeros(len(yy)); ok=np.zeros(len(yy))
            u=np.zeros(nc)
            if r is not None:
                xxall[valid]=rx; yyall[valid]=ry; ok[valid]=1
                u=np.bincount(cluster[valid],weights=rx*ry,minlength=nc)
                if verify:
                    rr=residual_pair(xx,yy,a,'scipy')[3]
                    assert math.isclose(rr,r,rel_tol=1e-7,abs_tol=1e-8),(h,inc,c,r,rr)
                    manual=[math.fsum(float(v) for v in (rx*ry)[cluster[valid]==k]) for k in range(nc)]
                    assert np.allclose(u,manual,rtol=1e-12,atol=1e-10)
            rows.append(row); arrays.append((xxall,yyall,ok)); scores.append(u)
    assert len(rows)==71
    u=np.column_stack(scores); denom=np.sqrt((u*u).sum(axis=0)); denom[denom<1e-14]=np.inf
    t=u.sum(axis=0)/denom
    rng=np.random.default_rng(SEED)
    signs=rng.choice([-1.,1.],size=(9999,nc))
    null=signs@u/denom
    maxnull=np.abs(null).max(axis=1)
    p=(1+(maxnull[:,None]>=np.abs(t)[None,:]).sum(axis=0))/10000
    # Cluster bootstrap by farmer, calendar blocks resampled with replacement.
    rng=np.random.default_rng(SEED+1)
    clfarm={f:np.unique(cluster[frame.farm.eq(f)]) for f in ['F13','F47']}
    boot=[]
    cluster_moments=[]
    for a,b,ok in arrays:
        cluster_moments.append(np.column_stack([np.bincount(cluster,weights=v,minlength=nc) for v in [ok,a,b,a*a,b*b,a*b]]))
    moments=np.stack(cluster_moments,axis=1) # cluster,test,6
    for _ in range(2000):
        sampled=np.concatenate([rng.choice(v,size=len(v),replace=True) for v in clfarm.values()])
        w=np.bincount(sampled,minlength=nc)
        m=np.tensordot(w,moments,axes=(0,0)); n=m[:,0]; n=np.where(n>0,n,1)
        cov=m[:,5]-m[:,1]*m[:,2]/n
        sd=np.sqrt(np.maximum(0,m[:,3]-m[:,1]**2/n)*np.maximum(0,m[:,4]-m[:,2]**2/n))
        boot.append(np.divide(cov,sd,out=np.full(71,np.nan),where=sd>1e-12))
    boot=np.array(boot)
    for j,row in enumerate(rows):
        v=boot[:,j]; v=v[np.isfinite(v)]
        row.update(score_t=float(t[j]),p_max=float(p[j]),ci_low=float(np.quantile(v,.025)) if len(v) else None,
                   ci_high=float(np.quantile(v,.975)) if len(v) else None)
        f1=row['r_F13']; f2=row['r_F47']; nohigh=row['r_without_high_ec']
        row['advance']=bool(row['p_max']<.01 and f1 is not None and f2 is not None and f1*f2>0 and min(abs(f1),abs(f2))>=.15 and nohigh is not None and rsign(row['r_partial'])*nohigh>0)
    if verify: ver=verify_features(x,fs)+['all_partial_correlations_scipy','all_cluster_scores_math']
    metadata={'days':len(frame),'rows':len(d),'cluster_count':nc,'tests':len(rows),'high_ec_days':int((yy>=1.2).sum()),
              'seed':SEED,'bootstrap':2000,'sign_flip':9999,'verification':ver,'advancing_tests':sum(r['advance'] for r in rows),
              'significant_max01':sum(r['p_max']<.01 for r in rows),'causal_feature_checks':'PASS' if verify else 'not_run',
              'model_training':False,'test_X_read':False,'locked_labels_read':False,
              'source_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['train_X.csv','train_y.csv']},
              'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version()}
    return rows,metadata,frame,fs

def rsign(v): return np.sign(v) if v is not None else 0

if __name__=='__main__':
    x,d=read(); rows,meta,frame,fs=analyze(x,d,verify=True)
    print('First computation and independent checks complete',flush=True)
    x2,d2=read(); rows2,meta2,_,_=analyze(x2,d2,verify=False)
    pd.testing.assert_frame_equal(pd.DataFrame(rows),pd.DataFrame(rows2))
    meta['rerun']='PASS'
    pd.DataFrame(rows).to_csv(OUT/'association_tests.csv',index=False,encoding='utf-8-sig')
    frame.to_csv(OUT/'daily_ec.csv',index=False,encoding='utf-8-sig')
    for h,z in fs.items(): z.to_csv(OUT/f'features_h{h}.csv',index=False,encoding='utf-8-sig')
    (OUT/'result.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    show=pd.DataFrame(rows).sort_values('p_max').head(12)
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    print(show[['hour','incremental','feature','n','r_partial','ci_low','ci_high','p_max','r_F13','r_F47','r_without_high_ec','advance']].to_string(index=False))
