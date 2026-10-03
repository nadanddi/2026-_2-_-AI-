from models_v3 import *
from run_models import metrics
def rawbag(fold,q):
 rows=[]
 for seed in range(1,5):
  z=np.load(R/f'집/코덱스/local/ec_dc4_integration_20261002_v1/DIAG10_{fold}_pfn_{seed}.npz');mp=dict(zip(z['row_id'].astype(str),z['raw_pfn']));rows.append([mp[i] for i in q.row_id])
 return np.mean(rows,axis=0)
def main():
 world=joblib.load(O/'world.joblib');effects=pd.read_csv(H/'effects.csv',float_precision='round_trip');out=[];maxdiff=0.;baseline=[];detail=[];core=S.loadcore();gm=json.loads((H/'ec_v3/EC_groups.json').read_text(encoding='utf-8'))
 for (fold,seed,farm,day),ee in effects[effects.target=='EC'].groupby(['fold','seed','farm','day']):
  lab=world['elab'];_,_,tm,vm=next(z for z in world['efolds'] if z[0]=='DIAG10' and z[1]==fold);tr,va=S.seasonal(lab[tm],lab[vm],world['wv']);q=va[(va.farm==farm)&(va.day==day)].sort_values('hour').reset_index(drop=True);model=joblib.load(O/f'ec_v3/EC_{fold}_{seed}_model.joblib');pfn=rawbag(fold,q);original=model.predict(q)['raw_r3'];base=finished_ec(.8*original+.2*pfn,tr,q,core);y=q.sub_ec.to_numpy();r=world['eouter'];r=r[(r.validator=='DIAG10')&(r.seed==seed)&(r.farm==farm)&(r.day==day)].set_index('row_id').loc[q.row_id];assert np.max(np.abs(base-r.season_v2.to_numpy()))<1e-7
  with S.threadpool_limits(limits=2):
   for e in ee.itertuples():
    donor=tr[(tr.farm==farm)&(tr.day==e.donor_day)].sort_values('hour');change=q.copy();cs=gm[e.group];change[cs]=donor[cs].to_numpy();pr=model.predict(change)['raw_r3'];pred=finished_ec(.8*pr+.2*pfn,tr,q,core);m=metrics(pred,y);new=e._asdict();new.pop('Index');new.update(m,mean_shift=float(np.mean(pred-base)),rmse_change=m['rmse']-metrics(base,y)['rmse']);maxdiff=max(maxdiff,abs(new['rmse_change']-e.rmse_change));out.append(new)
    if e.group in ['in_co2','in_temp'] and e.rank<=3 and (farm,int(day)) in EC[:2]:
     v=e.group
     for arm in ['midnight_only','hours1_23_only']:
      rr=q[['row_id']+core.RAW].copy();rr.loc[rr.index==0 if arm=='midnight_only' else rr.index>0,v]=donor[v].to_numpy()[:1] if arm=='midnight_only' else donor[v].to_numpy()[1:]
      changed=core.features(rr);changed['season']=q.season.to_numpy();pred=finished_ec(.8*model.predict(changed)['raw_r3']+.2*pfn,tr,q,core);mm=metrics(pred,y);detail.append(dict(seed=seed,farm=farm,day=int(day),group=v,arm=arm,donor_day=int(e.donor_day),rank=e.rank,mean_shift=float(np.mean(pred-base)),rmse_change=mm['rmse']-metrics(base,y)['rmse'],**mm))
 a=pd.concat([effects[effects.target=='TEMP'],pd.DataFrame(out)]);a.to_csv(H/'effects_v2.csv',index=False);pd.DataFrame(detail).to_csv(H/'ec_time_details.csv',index=False);savej(H/'ec_finish_audit.json',dict(status='PASS',old_separate_clip_max_rmse_difference=maxdiff,new_method='shrink/clip(.8 raw R3 + .2 raw PFNbag), original exact equation',n=len(out)))
 agg=a.groupby(['target','farm','day','group']).agg(mean_rmse_change=('rmse_change','mean'),median_rmse_change=('rmse_change','median'),min_rmse_change=('rmse_change','min'),max_rmse_change=('rmse_change','max'),mean_prediction_shift=('mean_shift','mean'),positive_fraction=('rmse_change',lambda v:float((v>0).mean())),n=('rmse_change','size')).reset_index();agg['rank']=agg.groupby(['target','farm','day']).mean_rmse_change.rank(ascending=False,method='min');agg.to_csv(H/'group_effect_summary_v2.csv',index=False)
 print('EXACT_EC',maxdiff);print(pd.DataFrame(detail).groupby(['farm','day','group','arm'])[['rmse_change','mean_shift']].mean().to_string())
if __name__=='__main__':main()
