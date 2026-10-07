from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import pandas as pd,numpy as np
S=H.parent/'ec_vent_gap_20261006_v1';R=H/'results_v1';O=H/'results_v2';O.mkdir(exist_ok=False)
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
pairs=pd.read_csv(R/'pairs.csv',float_precision='round_trip');p=pairs[(pairs['rank']==1)&(pairs['mode'].isin(['input','input_y']))].drop_duplicates(['farm','failed_day','success_day'])
rp=pd.read_csv(S/'results_v2/row_predictions.csv',float_precision='round_trip')
d=pd.read_csv(S/'results_v2/ordinary_days.csv',float_precision='round_trip')
all_d=pd.read_csv(S/'results_v2/days.csv',float_precision='round_trip')
raw=pd.read_csv(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].copy()
raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[8:10].astype(int)
allowed=set(zip(all_d.farm,all_d.day));raw=raw[[k in allowed for k in zip(raw.farm,raw.day)]]
cols=['in_temp','in_hum','in_co2','act_vent','act_thermal','act_shade','act_heating','act_circfan','act_co2','act_fog']
q=rp.merge(raw[['row_id']+cols],on='row_id',validate='one_to_one')
rows=[];gaps=[];trainctx=[];nearest=[]
targets=set(zip(p.farm,p.failed_day))|set(zip(p.farm,p.success_day))
q=q[[k in targets for k in zip(q.farm,q.day)]]
for (f,day),z in q.groupby(['farm','day']):
    z=z.sort_values('hour');info=d[(d.farm==f)&(d.day==day)].iloc[0]
    for h in [0,6,12,23]:
        u=z[z.hour<=h];a=z[z.hour==h].iloc[0]
        row=dict(farm=f,day=int(day),hour=h,fold=int(info.fold),true_y=float(a.y),prediction=float(a.baseline),prefix_y=float(u.y.mean()),prefix_prediction=float(u.baseline.mean()))
        for c in cols:row[c]=float(a[c]);row[c+'_prefix']=float(u[c].mean())
        for n in ['et','lgb','mlp','pfn']:row[n]=float(a[n+'_smooth']);row[n+'_prefix']=float(u[n+'_smooth'].mean())
        rows.append(row)
pd.DataFrame(rows).to_csv(O/'pair_prefix_all.csv',index=False)
for t in p.itertuples():
    for h in [0,6,12,23]:
        train_profile=raw[raw.hour<=h].groupby(['farm','day'])[cols].mean();sc=train_profile.std().replace(0,1)
        # Standardized prefix similarity across the same ten fields; diagnostic only.
        v=train_profile.loc[(t.farm,t.failed_day)];w=train_profile.loc[(t.farm,t.success_day)]
        dist=float(np.sqrt(np.mean(((v-w)/sc)**2)))
        gaps.append(dict(farm=t.farm,failed_day=int(t.failed_day),success_day=int(t.success_day),hour=h,prefix_distance=dist,day_distance=t.distance,failed_fold=int(t.failed_fold),success_fold=int(t.success_fold)))
pd.DataFrame(gaps).to_csv(O/'prefix_similarity.csv',index=False)
C=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/components'
for f,day in targets:
    info=d[(d.farm==f)&(d.day==day)].iloc[0];k=int(info.fold)
    with np.load(C/f'DIAG10_{k}_r3_7.npz',allow_pickle=False) as z:trainids=z['train_row_id'].astype(str)
    keys={(i[:3],int(i[4:7])) for i in trainids};tr=all_d[[x in keys for x in zip(all_d.farm,all_d.day)]];tr=tr[tr.farm==f]
    trainctx.append(dict(farm=f,day=int(day),fold=k,train_days=len(tr),train_high=int((tr.ymean>=1).sum()),train_mean_y=float(tr.ymean.mean()),train_closed=int(tr.closed.sum()),train_closed_high=int((tr.closed&(tr.ymean>=1)).sum())))
    for h in [0,6]:
        profiles=raw[raw.hour<=h].groupby(['farm','day'])[cols].mean()
        idx=[(r.farm,int(r.day)) for r in tr.itertuples()];x=profiles.loc[idx];sc=x.std().replace(0,1);md=x.median();v=profiles.loc[(f,day)].fillna(md)
        ds=np.mean(((x.fillna(md)-v)/sc)**2,axis=1);ix=ds.nsmallest(5).index;ts=tr.set_index(['farm','day']).loc[ix]
        nearest.append(dict(farm=f,day=int(day),hour=h,fold=k,nearest_distance=float(np.sqrt(ds.min())),neighbor_high=float((ts.ymean>=1).mean()),neighbor_y=float(ts.ymean.mean()),neighbor_days=';'.join(map(str,ts.index.get_level_values('day')))))
pd.DataFrame(trainctx).to_csv(O/'fold_training_context.csv',index=False);pd.DataFrame(nearest).to_csv(O/'prefix_training_neighbors.csv',index=False)
save('completion.json',dict(status='COMPLETE_PREFIX_DIAGNOSTIC',pairs=len(p),targets=len(targets),score_trace_rows=len(rows),fit=0,raw_y_reads=0,test_reads=0,actual_model_paths=False))
print('COMPLETE_PREFIX',len(p),len(targets))
