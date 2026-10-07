from pathlib import Path
import sys,json,math,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
S=H.parent/'ec_vent_gap_20261006_v1';O=H/'results_v1';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(n,x):pd.DataFrame(x).to_csv(O/n,index=False)
paths=[S/'results_v2/ordinary_days.csv',S/'results_v2/row_predictions.csv',S/'results_v2/days.csv']
save('registration.json',dict(source=sha(__file__),protocol=sha(H/'PROTOCOL.md'),inputs={str(p):sha(p) for p in paths},fit=0))
d=pd.read_csv(paths[0],float_precision='round_trip');rp=pd.read_csv(paths[1],float_precision='round_trip');all_d=pd.read_csv(paths[2],float_precision='round_trip')
d['fanlow']=d.act_circfan_mean<10;d['success']=d.mse<=.1**2;d['severe']=abs(d.bias)>=.2
cols=['in_temp_mean','in_hum_mean','in_co2_mean','act_vent_mean','act_thermal_mean','act_shade_mean','act_heating_mean','act_circfan_mean','act_co2_mean','act_fog_mean']
sc=d[cols].std().replace(0,1);md=d[cols].median();x=d[cols].fillna(md)
summ=[]
for name,mask in [('ordinary',d.ordinary),('closed',d.closed),('closed_fanlow',d.closed&d.fanlow),('closed_fanhigh',d.closed&~d.fanlow),('rest',~d.closed)]:
    q=d[mask];summ.append(dict(group=name,days=len(q),success=int(q.success.sum()),severe=int(q.severe.sum()),rmse=float(np.sqrt(q.mse.mean())),successful_rmse=float(np.sqrt(q.loc[q.success,'mse'].mean()))))
table('group_success.csv',summ)
pairs=[];coverage=[]
for ix,f in d[d.closed&d.fanlow&d.severe].iterrows():
    base=d[(d.farm==f.farm)&(d.phase==f.phase)&d.closed&d.fanlow&d.success&(abs(d.in_temp_mean-f.in_temp_mean)<=2)&(abs(d.in_hum_mean-f.in_hum_mean)<=10)]
    for mode in ['input','input_y','same_fold']:
        c=base.copy()
        if mode=='input_y':c=c[abs(c.ymean-f.ymean)<=.1]
        if mode=='same_fold':c=c[c.fold==f.fold]
        coverage.append(dict(farm=f.farm,day=int(f.day),mode=mode,success_candidates=len(c)))
        if not len(c):continue
        dist=np.mean(((x.loc[c.index]-x.loc[ix])/sc)**2,axis=1)
        for rank,(j,v) in enumerate(dist.nsmallest(3).items(),1):
            s=d.loc[j];p=dict(farm=f.farm,failed_day=int(f.day),success_day=int(s.day),mode=mode,rank=rank,distance=float(np.sqrt(v)),failed_fold=int(f.fold),success_fold=int(s.fold),failed_y=f.ymean,success_y=s.ymean,failed_p=f.pmean,success_p=s.pmean,failed_bias=f.bias,success_bias=s.bias,failed_rmse=float(np.sqrt(f.mse)),success_rmse=float(np.sqrt(s.mse)))
            for col in cols+['ventzero']:
                p['failed_'+col]=float(f[col]);p['success_'+col]=float(s[col]);p['delta_'+col]=float(f[col]-s[col])
            pairs.append(p)
table('pairs.csv',pairs);table('pair_coverage.csv',coverage)
table('successful_closed_fanlow.csv',d[d.closed&d.fanlow&d.success])
weights={'et':.48,'lgb':.24,'mlp':.08,'pfn':.20};members=[]
for (farm,day),q in rp.groupby(['farm','day']):
    y=float(q.y.mean());p=float(q.baseline.mean());out=dict(farm=farm,day=int(day),ymean=y,pmean=p,bias=p-y)
    for n,w in weights.items():
        val=float(q[n+'_smooth'].mean());out[n]=val;out[n+'_weighted_bias']=w*(val-y)
    assert abs(sum(out[n+'_weighted_bias'] for n in weights)-(p-y))<1e-12
    out['raw_mix']=float(q.raw_mix.mean());out['smooth_mix']=float(q.smooth_mix.mean());out['smoothing_shift']=out['smooth_mix']-out['raw_mix'];members.append(out)
table('day_member_scores.csv',members)
ps=pd.DataFrame(pairs);targets={('F47',130),('F47',161),('F13',132),('F13',231),('F13',233),('F13',156),('F47',216)}
for r in ps[ps['rank']==1].itertuples():targets.add((r.farm,r.success_day))
pred=rp[[t in targets for t in zip(rp.farm,rp.day)]].copy()
raw=pd.read_csv(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.isin(pred.row_id)]
pred=pred.merge(raw,on='row_id',validate='one_to_one')
rows=[];prefix=[]
for (farm,day),q in pred.groupby(['farm','day']):
    q=q.sort_values('hour');ss=d[(d.farm==farm)&(d.day==day)].iloc[0]
    for h in [0,6,12,23]:
        r=q[q.hour==h].iloc[0];out=dict(farm=farm,day=int(day),hour=h,fold=int(ss.fold),true_y=float(r.y),pred=float(r.baseline),success=bool(ss.success),daily_rmse=float(np.sqrt(ss.mse)),prefix_y=float(q[q.hour<=h].y.mean()),prefix_p=float(q[q.hour<=h].baseline.mean()))
        for c in cols:
            col=c.removesuffix('_mean');out[col]=float(r[col]);out[col+'_prefix']=float(q.loc[q.hour<=h,col].mean())
        for n in weights:out[n]=float(r[n+'_smooth']);out[n+'_prefix']=float(q.loc[q.hour<=h,n+'_smooth'].mean())
        rows.append(out)
table('paired_prefix_profiles.csv',rows)
table('selected_hour_rows.csv',pred)
save('completion.json',dict(status='COMPLETE_DESCRIPTIVE',ordinary=329,closed=61,closed_fanlow=50,pairs=len(pairs),severe_closedfan=int((d.closed&d.fanlow&d.severe).sum()),success_closedfan=int((d.closed&d.fanlow&d.success).sum()),fit=0,test_reads=0,EL1_reads=0,lock_reads=0))
print(pd.DataFrame(summ).to_string(index=False));print(ps[ps['rank']==1][['farm','failed_day','success_day','mode','distance','failed_y','success_y','failed_p','success_p']].to_string(index=False));print('COMPLETE')
