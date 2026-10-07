from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
S=H.parent/'ec_vent_gap_20261006_v1';O=H/'results_v4';O.mkdir(exist_ok=False)
d=pd.read_csv(S/'results_v2/ordinary_days.csv',float_precision='round_trip')
rp=pd.read_csv(S/'results_v2/row_predictions.csv',float_precision='round_trip')
raw=pd.read_csv(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.isin(rp.row_id)].copy()
raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[8:10].astype(int)
cols=['in_temp','in_hum','in_co2','act_vent','act_thermal','act_shade','act_heating','act_circfan','act_co2','act_fog']
# One fixed ordinary329 daily mean scale for all prefix times; no comparison across changing scales.
scale=d[[c+'_mean' for c in cols]].std().to_numpy().copy();scale[scale==0]=1
p=pd.read_csv(H/'results_v1/pairs.csv',float_precision='round_trip');p=p[(p['rank']==1)&p['mode'].isin(['input','input_y'])].drop_duplicates(['farm','failed_day','success_day'])
rows=[]
for r in p.itertuples():
    for h in [0,6,12,23]:
        a=raw[(raw.farm==r.farm)&(raw.day==r.failed_day)&(raw.hour<=h)][cols].mean().to_numpy()
        b=raw[(raw.farm==r.farm)&(raw.day==r.success_day)&(raw.hour<=h)][cols].mean().to_numpy()
        rows.append(dict(farm=r.farm,failed_day=int(r.failed_day),success_day=int(r.success_day),hour=h,distance_same_fixed_scale=float(np.sqrt(np.mean(((a-b)/scale)**2)))))
pd.DataFrame(rows).to_csv(O/'fixed_scale_prefix_similarity.csv',index=False)
# Additional exploratory 0h analogy: pool is same-farm/phase/closed/fanlow successful days.
base=d[(d.closed)&(d.act_circfan_mean<10)].copy();base['success']=base.mse<=.01
r0=rp[rp.hour==0].merge(raw[['row_id']+cols],on='row_id',validate='one_to_one').merge(base[['farm','day','phase','success','bias','mse']],on=['farm','day'],how='inner',validate='one_to_one')
rows=[]
for r in r0[abs(r0.bias)>=.2].itertuples():
    c=r0[(r0.farm==r.farm)&(r0.phase==r.phase)&r0.success&(abs(r0.baseline-r0.y)<=.1)]
    if not len(c):continue
    rv=np.array([getattr(r,col) for col in cols]);ds=np.mean(((c[cols].to_numpy()-rv)/scale)**2,axis=1)
    order=np.argsort(ds)[:3]
    for rank,i in enumerate(order,1):
        s=c.iloc[i];row=dict(farm=r.farm,failed_day=int(r.day),success_day=int(s.day),rank=rank,distance=float(np.sqrt(ds[i])),failed_y0=float(r.y),failed_p0=float(r.baseline),success_y0=float(s.y),success_p0=float(s.baseline),failed_day_rmse=float(np.sqrt(r.mse)),success_day_rmse=float(np.sqrt(s.mse)))
        for col in cols:row['failed_'+col]=float(getattr(r,col));row['success_'+col]=float(s[col])
        rows.append(row)
pd.DataFrame(rows).to_csv(O/'h0_success_analogies.csv',index=False)
(O/'completion.json').write_text(json.dumps(dict(status='COMPLETE_DIAGNOSTIC',same_scale='std of daily means across ordinary329 fixed for all times',h0_analogies=len(rows),posthoc=True,fit=0),indent=2),encoding='utf-8')
print(pd.DataFrame(rows)[pd.DataFrame(rows)['rank']==1][['farm','failed_day','success_day','distance','failed_y0','failed_p0','success_y0','success_p0']].to_string(index=False))
