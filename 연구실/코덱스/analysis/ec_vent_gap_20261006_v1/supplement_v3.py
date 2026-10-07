from pathlib import Path
import sys,json,hashlib,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
R=H/'results_v2';O=H/'results_v3';O.mkdir(exist_ok=False)
L=ROOT/'연구실/코덱스/local'/H.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(n,x):pd.DataFrame(x).to_csv(O/n,index=False)
save('registration.json',dict(source=sha(__file__),purpose='critic_feedback_and_fixed_descriptive_extensions_no_model',read_paths=[str(R),str(L/'inputs')]))
d=pd.read_csv(R/'days.csv',float_precision='round_trip');b=pd.read_csv(R/'ordinary_days.csv',float_precision='round_trip')
d['fanlow']=d.act_circfan_mean<10
def stat(q):
    return dict(n=len(q),high=int((q.ymean>=1).sum()),mean_y=float(q.ymean.mean()),median_y=float(q.ymean.median()),rmse=float(np.sqrt(q.sse.sum()/q.n.sum())),bias=float(q.bias.mean()),true_falsehigh=int(((q.ymean<1)&(q.pmean>=1)).sum()),wide_falsehigh=int(((q.ymean<1)&(q.pmean>=.9)).sum()),level=float(q.sse_day.sum()/q.sse.sum()),sse=float(q.sse.sum()))
state=[]
for keys,q in d.groupby(['closed','fanlow']):state.append(dict(closed=bool(keys[0]),fanlow=bool(keys[1]),**stat(q)))
for keys,q in d[d.ordinary].groupby(['closed','fanlow']):state.append(dict(ordinary=True,closed=bool(keys[0]),fanlow=bool(keys[1]),**stat(q)))
table('state_interaction.csv',state)
tri=[]
for keys,q in d[d.ordinary].groupby(['farm','phase','closed','fanlow']):tri.append(dict(farm=keys[0],phase=int(keys[1]),closed=bool(keys[2]),fanlow=bool(keys[3]),**stat(q)))
table('state_farm_phase.csv',tri)
c=b[b.closed];r=b[~b.closed];gap=float(c.mse.mean()-r.mse.mean())
parts={'constant_bias':float(c.bias.mean()**2-r.bias.mean()**2),'daily_bias_variance':float(c.bias.var(ddof=0)-r.bias.var(ddof=0)),'within_day_shape':float(c.sse_shape.sum()/c.n.sum()-r.sse_shape.sum()/r.n.sum())}
assert abs(sum(parts.values())-gap)<1e-12
save('gap_budget.json',dict(gap_mse=gap,parts=parts,fractions={k:v/gap for k,v in parts.items()}))
trim=[]
for gr,q in b.groupby('group'):
    q=q.sort_values(['sse','farm','day'],ascending=[False,True,True])
    for frac in [0,.05,.1,.2]:
        k=math.floor(len(q)*frac);u=q.iloc[k:];trim.append(dict(group=gr,fraction=frac,removed=k,removed_share=float(q.iloc[:k].sse.sum()/q.sse.sum()),**stat(u)))
table('equal_fraction_trim.csv',trim)
z=pd.read_csv(L/'inputs/oof.csv',float_precision='round_trip');z=z[z.validator=='DIAG10']
seedrows=[];cache=[];neighbors=[]
rawcols=['in_temp_mean','in_hum_mean','in_co2_mean','act_vent_mean','act_heating_mean','act_circfan_mean','act_shade_mean','act_thermal_mean','act_co2_mean','act_fog_mean']
C=L/'inputs/components'
for s,q in z.groupby('seed'):
    q=q.merge(b[['farm','day','group']],on=['farm','day'],how='inner',validate='many_to_one')
    for gr,u in q.groupby('group'):
        e=u.baseline-u.y;ed=e.groupby([u.farm,u.day]).transform('mean')
        seedrows.append(dict(seed=int(s),group=gr,days=len(u)//24,rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),level=float(np.dot(ed,ed)/np.dot(e,e))))
table('seed_check.csv',seedrows)
for k in range(10):
    trainsets=[]
    for s in [7,101,2024]:
        p=C/f'DIAG10_{k}_r3_{s}.npz';assert p.exists()
        with np.load(p,allow_pickle=False) as zz:
            ids=zz['row_id'].astype(str);train=zz['train_row_id'].astype(str);trainsets.append(set(train))
            qq=z[(z.fold==k)&(z.seed==s)].set_index('row_id').loc[ids]
            for n in ['raw_et','raw_lgb','raw_mlp']:assert np.max(abs(qq[n]-zz[n]))<1e-12
            assert np.max(abs(qq.y-zz['sub_ec']))<1e-12
            cache.append(dict(path=str(p),sha=sha(p),rows=len(ids),train_rows=len(train)))
    assert trainsets[0]==trainsets[1]==trainsets[2]
    trkeys={(i[:3],int(i[4:7])) for i in trainsets[0]}
    assert len(trainsets[0])==24*len(trkeys)
    tr=d[[tuple(t) in trkeys for t in zip(d.farm,d.day)]]
    availablekeys=set(zip(tr.farm,tr.day));assert availablekeys==trkeys,'Partial fold training support must be documented, never silently assumed complete'
    # All fold training days in this recipe are in unlocked360 publicOOF registry.
    for ix,q in b[b.fold==k].iterrows():
        ts=tr[tr.farm==q.farm];ids=set(zip(ts.farm,ts.day));assert (q.farm,q.day) not in trkeys
        assert min(abs(ts.day-q.day))>=2
        sc=ts[rawcols].std().replace(0,1);med=ts[rawcols].median();ds=np.mean(((ts[rawcols].fillna(med)-q[rawcols].fillna(med).astype(float))/sc)**2,axis=1)
        ns=ts.loc[ds.nsmallest(5).index]
        nearest=ns.iloc[0]
        neighbors.append(dict(farm=q.farm,day=int(q.day),group=q.group,fold=int(k),train_days_total=len(trkeys),available_days=len(tr),samefarm_train=len(ts),nearest_distance=float(np.sqrt(ds.min())),nearest_y=float(nearest.ymean),nearest_day=int(nearest.day),neighbor_y=float(ns.ymean.mean()),neighbor_high=float((ns.ymean>=1).mean()),ymean=q.ymean,pmean=q.pmean,bias=q.bias,fanlow=bool(q.act_circfan_mean<10)))
table('training_neighbors_verified.csv',neighbors);save('cache_checks.json',dict(status='PASS',r3_caches=30,all_fold_train_days_covered=True,files=cache))
nn=pd.DataFrame(neighbors);nr=[]
for facets in [['group'],['group','fanlow']]:
    for key,q in nn.groupby(facets):
        nr.append(dict(facets='/'.join(facets),key=str(key),n=len(q),neighbor_high_mean=float(q.neighbor_high.mean()),neighbor_y_mean=float(q.neighbor_y.mean()),nearest_y_mean=float(q.nearest_y.mean()),median_distance=float(q.nearest_distance.median()),true_y=float(q.ymean.mean())))
table('neighbor_summary.csv',nr)
table('falsehigh_neighbors.csv',nn[nn.pmean>=.9].sort_values('bias',ascending=False))
rp=pd.read_csv(R/'row_predictions.csv',float_precision='round_trip')
allraw=pd.read_csv(Path(env.DATA)/'train_X.csv');allraw=allraw[allraw.row_id.isin(rp.row_id)]
ar=rp.merge(allraw,on='row_id',validate='one_to_one')
patterns=[]
for gr,q in ar.groupby('group'):
    for c in [x for x in allraw if x!='row_id' and allraw[x].notna().any()]:
        for h in range(24):
            t=q[q.hour==h];patterns.append(dict(group=gr,channel=c,hour=h,mean=float(t[c].mean()),missing=int(t[c].isna().sum()),zero=float((t[c]==0).mean())))
table('input_hour_profiles.csv',patterns)
onset=[]
for (f,day),q in rp.groupby(['farm','day']):
    q=q.sort_values('hour');prefix=q.baseline.expanding().mean();ys=q.y.mean()
    for h in [0,6,12,23]:onset.append(dict(farm=f,day=int(day),group=q.group.iloc[0],hour=h,ymean=float(ys),prefix=float(prefix.iloc[h]),selected09=bool(prefix.iloc[h]>=.9),selected1=bool(prefix.iloc[h]>=1),rmse=float(np.sqrt(np.mean((q.baseline.iloc[:h+1]-q.y.iloc[:h+1])**2)))))
table('prefix_cases.csv',onset)
case=[]
for t in b.sort_values('sse',ascending=False).head(15).itertuples():
    q=rp[(rp.farm==t.farm)&(rp.day==t.day)]
    case.append(dict(farm=t.farm,day=int(t.day),group=t.group,ymean=t.ymean,pmean=t.pmean,bias=t.bias,sse=t.sse,et=float(q.et_smooth.mean()),lgb=float(q.lgb_smooth.mean()),mlp=float(q.mlp_smooth.mean()),pfn=float(q.pfn_smooth.mean()),ventzero=t.ventzero,fan=t.act_circfan_mean,temp=t.in_temp_mean,hum=t.in_hum_mean,heat=t.act_heating_mean,co2=t.in_co2_mean))
table('top_case_members.csv',case)
save('completion.json',dict(status='PASS_DESCRIPTIVE_EXTENSION',ordinary=len(b),raw_caches=30,train_coverage='All DIAG fold train days found in public360; no raw train_y read',fit=0,test_reads=0,EL1_reads=0))
print('COMPLETE_V3',json.dumps(parts),flush=True)
