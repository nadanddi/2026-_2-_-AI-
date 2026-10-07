from pathlib import Path
import sys,csv,json,math,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research')); import env
import numpy as np,pandas as pd
L=ROOT/'연구실/코덱스/local'/H.name
OUT=H/'results_v2'; OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(name,x):pd.DataFrame(x).to_csv(OUT/name,index=False)
def finite(x):return float(x) if np.isfinite(x) else None
oof=L/'inputs/oof.csv'; raw=Path(env.DATA)/'train_X.csv'
old=ROOT/'집/코덱스/analysis/ec_daily_level_audit_20261004_v1/daily_v3.csv'
save('registration.json',dict(source_sha=sha(__file__),protocol_sha=sha(H/'PROTOCOL.md'),inputs={str(p):sha(p) for p in [oof,raw,old]},fit=0,test_reads=0,raw_y_reads=0))
proof=json.loads((ROOT/'집/코덱스/analysis/ec_lgb_operation_20261004_v1/full_verification_v3.json').read_text(encoding='utf-8'))
assert sha(oof)==proof['aggregate_sha256']
z=pd.read_csv(oof,float_precision='round_trip'); z=z[z.validator=='DIAG10'].copy()
assert len(z)==25920 and set(z.seed)=={7,101,2024}
keys=['row_id','farm','day','hour','fold']
assert (z.groupby('row_id').y.nunique()==1).all()
a=z.groupby(keys,as_index=False)[['y','baseline','raw_et','raw_lgb','raw_mlp','old_pfn_raw','clip_lo','clip_hi']].mean()
a=a.sort_values(['farm','day','hour']).reset_index(drop=True);assert len(a)==8640
x=pd.read_csv(raw);x=x[x.row_id.isin(a.row_id)].copy();assert len(x)==8640
channels=[c for c in x if c!='row_id' and x[c].notna().any()]
assert len(channels)==14
a=a.merge(x,on='row_id',validate='one_to_one');a['error']=a.baseline-a.y
g=a.groupby(['farm','day'],sort=True)
d=g.agg(n=('y','size'),fold=('fold','first'),ymean=('y','mean'),ystd=('y','std'),ymin=('y','min'),ymax=('y','max'),pmean=('baseline','mean'),bias=('error','mean'))
d['sse']=g.error.apply(lambda s:float(np.dot(s,s)));d['mae']=g.error.apply(lambda s:float(abs(s).mean()))
d['sse_day']=d.n*d.bias**2;d['sse_shape']=d.sse-d.sse_day;d['mse']=d.sse/d.n
d['ventzero']=g.act_vent.apply(lambda s:float((s==0).mean()));d['closed']=d.ventzero>=.8
d['ordinary']=d.ymean<1;d['phase']=(d.index.get_level_values('day')>=179).astype(int)
for c in channels:
    for name,fn in [('mean','mean'),('std','std'),('min','min'),('max','max')]:d[c+'_'+name]=getattr(g[c],fn)()
    d[c+'_missing']=g[c].apply(lambda s:float(s.isna().mean()))
    for t,m in [('night',a.hour<=5),('daytime',a.hour.between(10,15))]:d[c+'_'+t]=a[m].groupby(['farm','day'])[c].mean()
    if c.startswith('act_'):
        d[c+'_zero']=g[c].apply(lambda s:float((s==0).mean()))
        d[c+'_switch']=g[c].apply(lambda s:float((s.diff().iloc[1:]!=0).sum()))
d=d.reset_index();assert set(d.n)=={24} and len(d)==360
assert np.allclose(d.sse,d.sse_day+d.sse_shape,rtol=0,atol=1e-12)
prior=pd.read_csv(old,float_precision='round_trip');prior=prior[(prior.model=='baseline')&(prior.seed=='mean')]
merged=d.merge(prior,on=['farm','day'],suffixes=('','_old'),validate='one_to_one')
for c,oc in [('sse','sse'),('sse_day','sse_day'),('bias','bias'),('ymean','ymean'),('ventzero','sealed_fraction')]:assert np.max(abs(merged[c]-merged[oc+'_old' if oc+'_old' in merged.columns else oc]))<1e-11
b=d[d.ordinary].copy();assert len(b)==329 and b.closed.sum()==61
b['group']=np.where(b.closed,'closed','rest')
b['yband']=pd.cut(b.ymean,[-np.inf,.2,.4,.6,.8,1],right=False).astype(str)
b['tempband']=pd.cut(b.in_temp_mean,[-np.inf,12,14,16,18,np.inf],right=False).astype(str)
b['humband']=pd.cut(b.in_hum_mean,[-np.inf,60,75,85,np.inf],right=False).astype(str)
b['heatband']=np.where(b.act_heating_mean>0,'heat','noheat')
b['fanband']=np.where(b.act_circfan_mean<10,'fan<10','fan>=10')
b['type']=np.select([b.pmean>=.9,b.bias>.1,b.bias<-.1],['falsehigh','over','under'],default='within.1')
def summary(q):
    return dict(days=len(q),rmse=float(np.sqrt(q.sse.sum()/q.n.sum())),mae=float(q.mae.mean()),bias=float(q.bias.mean()),median_abs_bias=float(abs(q.bias).median()),day_bias_q90=float(abs(q.bias).quantile(.9)),day_bias_q95=float(abs(q.bias).quantile(.95)),ymean=float(q.ymean.mean()),pmean=float(q.pmean.mean()),ystd=float(q.ystd.mean()),sse=float(q.sse.sum()),day_share=float(q.sse_day.sum()/q.sse.sum()),shape_mse=float(q.sse_shape.sum()/q.n.sum()),level_mse=float(q.sse_day.sum()/q.n.sum()),constant_bias_mse=float(q.bias.mean()**2),level_variance=float(q.bias.var(ddof=0)),within01=int((abs(q.bias)<=.1).sum()),falsehigh=int((q.pmean>=.9).sum()))
summ={s:summary(q) for s,q in b.groupby('group')}
save('summary.json',summ);table('days.csv',d);table('ordinary_days.csv',b)
rows=[]
for c in [q for q in b if any(q.endswith(s) for s in ['_mean','_std','_min','_max','_night','_daytime','_missing','_zero','_switch'])]+['ymean','ystd','pmean','bias','mae','ventzero']:
    u=b.loc[b.closed,c].dropna();v=b.loc[~b.closed,c].dropna();den=np.sqrt((u.var(ddof=1)+v.var(ddof=1))/2)
    rows.append(dict(feature=c,closed_n=len(u),rest_n=len(v),closed_mean=float(u.mean()),rest_mean=float(v.mean()),closed_median=float(u.median()),rest_median=float(v.median()),smd=finite((u.mean()-v.mean())/den),closed_q10=float(u.quantile(.1)),closed_q90=float(u.quantile(.9)),rest_q10=float(v.quantile(.1)),rest_q90=float(v.quantile(.9))))
table('feature_comparison.csv',rows)
rows=[];decomps=[]
for facets in [['farm'],['phase'],['fold'],['yband'],['tempband'],['humband'],['fanband'],['heatband'],['farm','phase'],['farm','yband'],['farm','tempband'],['farm','phase','yband']]:
    f='/'.join(facets); common=[]
    for key,q in b.groupby(facets,observed=True):
        key=key if isinstance(key,tuple) else (key,)
        for gr,u in q.groupby('group'):rows.append(dict(facets=f,stratum=str(key),group=gr,**summary(u)))
        if set(q.group)=={'closed','rest'}:
            c=q[q.closed];r=q[~q.closed];common.append(dict(key=str(key),nc=len(c),nr=len(r),mc=float(c.mse.mean()),mr=float(r.mse.mean())))
    if common:
        nc=sum(t['nc'] for t in common);nr=sum(t['nr'] for t in common)
        composition=sum((t['nc']/nc-t['nr']/nr)*(t['mc']+t['mr'])/2 for t in common)
        within=sum((t['nc']/nc+t['nr']/nr)/2*(t['mc']-t['mr']) for t in common)
        closed_mse=sum(t['nc']/nc*t['mc'] for t in common);rest_mse=sum(t['nr']/nr*t['mr'] for t in common)
        assert abs(composition+within-(closed_mse-rest_mse))<1e-12
        decomps.append(dict(facets=f,common_closed=nc,common_rest=nr,closed_mse=closed_mse,rest_mse=rest_mse,composition=composition,within=within,common_strata=len(common),rest_weighted_closed_mse=sum(t['nr']/nr*t['mc'] for t in common),strata=common))
table('strata.csv',rows);save('composition_decomposition.json',decomps)
types=[]
for (gr,t),q in b.groupby(['group','type']):types.append(dict(group=gr,type=t,group_sse_share=float(q.sse.sum()/b.loc[b.group==gr,'sse'].sum()),**summary(q)))
table('error_types.csv',types)
trim=[]
for gr,q in b.groupby('group'):
    q=q.sort_values(['sse','farm','day'],ascending=[False,True,True]);table(f'top_{gr}.csv',q)
    for k in [0,1,3,5,10]:
        u=q.iloc[k:];trim.append(dict(group=gr,removed=k,removed_ids=[f'{r.farm}_{r.day}' for r in q.iloc[:k].itertuples()],removed_sse_share=float(q.iloc[:k].sse.sum()/q.sse.sum()),**summary(u)))
save('trim_sensitivity.json',trim)
thresholds=[]
for t in [.5,.65,.8,.85,.9,1.0]:
    for gr,q in b.groupby(b.ventzero>=t):thresholds.append(dict(threshold=t,closed=bool(gr),**summary(q)))
table('threshold_sensitivity.csv',thresholds)
ordinarykeys=set(zip(b.farm,b.day));row=a[[tuple(z) in ordinarykeys for z in zip(a.farm,a.day)]].copy()
row=row.merge(b[['farm','day','group']],on=['farm','day'],validate='many_to_one')
def smooth(v,df):
    return .5*v+.5*pd.Series(v,index=df.index).groupby([df.farm,df.day]).transform(lambda s:s.expanding().mean()).to_numpy()
members={'et':('raw_et',.48),'lgb':('raw_lgb',.24),'mlp':('raw_mlp',.08),'pfn':('old_pfn_raw',.2)}
# Reconstruct each seed before averaging: clipping is nonlinear.
recon=[];components=[]
for s,q in z.groupby('seed'):
    q=q.sort_values(['farm','day','hour']).reset_index(drop=True)
    r=sum(q[col].to_numpy()*w for col,w in members.values());p=smooth(r,q);p=np.clip(p,q.clip_lo,q.clip_hi)
    assert np.max(abs(p-q.baseline))<1e-11
    c=q[['row_id','farm','day','hour','y','baseline']].copy();c['raw_mix']=r;c['smooth_mix']=smooth(r,q)
    for n,(col,w) in members.items():c[n]=q[col];c[n+'_smooth']=smooth(q[col].to_numpy(),q)
    c['seed']=s;recon.append(c)
rr=pd.concat(recon).groupby(['row_id','farm','day','hour'],as_index=False).mean(numeric_only=True)
rr=rr.merge(b[['farm','day','group']],on=['farm','day'],how='inner',validate='many_to_one')
for gr,q in rr.groupby('group'):
    for col in list(members)+[n+'_smooth' for n in members]+['raw_mix','smooth_mix','baseline']:
        e=q[col]-q.y;components.append(dict(group=gr,member=col,rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean())))
table('member_errors.csv',components);table('row_predictions.csv',rr)
hour=[]
for (gr,h),q in rr.groupby(['group','hour']):
    e=q.baseline-q.y;hour.append(dict(group=gr,hour=int(h),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),ymean=float(q.y.mean()),pmean=float(q.baseline.mean())))
table('hour_errors.csv',hour)
attribution=[]
for gr,q in rr.groupby('group'):
    emix=q.smooth_mix-q.y
    for n,(col,w) in members.items():attribution.append(dict(group=gr,member=n,weight=w,sse_allocation=float(w*np.dot(q[n+'_smooth']-q.y,emix))))
    assert abs(sum(t['sse_allocation'] for t in attribution if t['group']==gr)-np.dot(emix,emix))<1e-9
table('member_sse_allocation.csv',attribution)
# Same-farm/pass matching controls. Outcome is used only as a retrospective control.
matches=[]
for kind in ['input','input_y']:
    cols=['in_temp_mean','in_hum_mean','in_co2_mean','act_heating_mean','act_circfan_mean','act_shade_mean','act_thermal_mean','out_temp_mean']
    scale=b[cols].std().replace(0,1);median=b[cols].median();vals=b[cols].fillna(median)
    for ix,c in b[b.closed].iterrows():
        candidates=b[(~b.closed)&(b.farm==c.farm)&(b.phase==c.phase)&(abs(b.in_temp_mean-c.in_temp_mean)<=2)]
        if kind=='input_y':candidates=candidates[abs(candidates.ymean-c.ymean)<=.1]
        if not len(candidates):continue
        dist=np.mean(((vals.loc[candidates.index]-vals.loc[ix])/scale)**2,axis=1)
        j=dist.idxmin();r=b.loc[j]
        matches.append(dict(kind=kind,farm=c.farm,closed_day=int(c.day),rest_day=int(r.day),distance=float(np.sqrt(dist[j])),closed_y=c.ymean,rest_y=r.ymean,closed_bias=c.bias,rest_bias=r.bias,closed_mse=c.mse,rest_mse=r.mse,closed_temp=c.in_temp_mean,rest_temp=r.in_temp_mean))
table('matched_pairs.csv',matches)
matching=[]
for kind,q in pd.DataFrame(matches).groupby('kind'):matching.append(dict(kind=kind,pairs=len(q),unique_rest=len(q[['farm','rest_day']].drop_duplicates()),closed_rmse=float(np.sqrt(q.closed_mse.mean())),rest_rmse=float(np.sqrt(q.rest_mse.mean())),closed_bias=float(q.closed_bias.mean()),rest_bias=float(q.rest_bias.mean()),mean_abs_ydiff=float(abs(q.closed_y-q.rest_y).mean()),median_distance=float(q.distance.median())))
save('matching_summary.json',matching)
# Block uncertainty: fixed ordinary eligibility, contiguous observed public dates per farm.
blocks=[]
for f,q in d.groupby('farm'):
    q=q.sort_values('day')
    for start in range(0,len(q),5):blocks.append(q.iloc[start:start+5].index.to_numpy())
rng=np.random.default_rng(20261006);draws=[]
for _ in range(2000):
    idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]);q=d.loc[idx];q=q[q.ordinary]
    c=q[q.closed];r=q[~q.closed]
    if len(c) and len(r):draws.append([np.sqrt(c.mse.mean())-np.sqrt(r.mse.mean()),c.bias.mean()-r.bias.mean()])
save('bootstrap.json',dict(blocks=len(blocks),draws=len(draws),method='5 successive public records within farm; pooled blocks across farms; retrospective not causal',gap_rmse_ci95=np.quantile(np.array(draws)[:,0],[.025,.975]).tolist(),gap_bias_ci95=np.quantile(np.array(draws)[:,1],[.025,.975]).tolist()))
# Fold-training neighborhoods: illustrative similarity, not learned model attribution.
neighbors=[];C=L/'inputs/components'
for k in range(10):
    p=C/f'DIAG10_{k}_r3_7.npz'
    if not p.exists():continue
    with np.load(p,allow_pickle=False) as zz:trainids=zz['train_row_id'].astype(str)
    trkeys={(r[:3],int(r[4:7])) for r in trainids};tr=d[[tuple(t) in trkeys for t in zip(d.farm,d.day)]]
    q=b[b.fold==k]
    assert not set(zip(q.farm,q.day))&trkeys
    cols=['in_temp_mean','in_hum_mean','in_co2_mean','act_vent_mean','act_heating_mean','act_circfan_mean','act_shade_mean','act_thermal_mean','act_co2_mean','act_fog_mean']
    for ix,r in q.iterrows():
        candidates=tr[tr.farm==r.farm];med=candidates[cols].median();sc=candidates[cols].std().replace(0,1)
        vals=candidates[cols].fillna(med);query=r[cols].fillna(med).astype(float)
        ds=np.mean(((vals-query)/sc)**2,axis=1);top=ds.nsmallest(5);ts=candidates.loc[top.index]
        neighbors.append(dict(farm=r.farm,day=int(r.day),group=r.group,fold=int(k),ymean=r.ymean,pmean=r.pmean,nearest_distance=float(np.sqrt(top.iloc[0])),neighbor_y_mean=float(ts.ymean.mean()),neighbor_high_fraction=float((ts.ymean>=1).mean()),neighbor_days=';'.join(map(str,ts.day)),sse=r.sse))
table('training_neighbors.csv',neighbors)
save('completion.json',dict(status='COMPLETE_DESCRIPTIVE',rows=len(a),ordinary=329,closed=61,rest=268,neighbors=len(neighbors),max_old_daily_gap=float(np.max(abs(merged.sse-merged.sse_old))),fit=0,test_reads=0,lock_y_reads=0,adoption=0))
print(json.dumps(summ,ensure_ascii=False,indent=2));print('COMPLETE',OUT,flush=True)
