from prepare import *
def main():
 raw=pd.read_csv(O/'temperature_raw_public.csv',float_precision='round_trip');events=pd.read_csv(R/'집/코덱스/analysis/success_signals_20261003_v1/event_days.csv');weather=['out_temp','out_hum','out_rad','out_wspd'];ds={k:g.sort_values('hour') for k,g in raw.groupby(['farm','day'])};overlaps=[];members=[];summaries=[]
 w=joblib.load(O/'world.joblib')
 for event in events[events.good].itertuples():
  key=(event.farm,int(event.day));q=ds[key];fold=event.fold
  if event.target=='TEMP':
   fd=next(fd for name,k,fd in w['folds'] if name=='DIAG10' and k==fold);tm,vm=S.common.split_mask(w['lab'],fd);keys=set(w['lab'][tm][['farm','day']].itertuples(index=False,name=None))
   oo=w['outer'];oo=oo[(oo.validator=='DIAG10')&(oo.base_seed==7)&(oo.context=='1-8')&(oo.farm==event.farm)&(oo.day==event.day)]
   for member,z in oo.groupby('member'):
    e=z.prediction-z.sub_temp;members.append(dict(target='TEMP',farm=event.farm,day=event.day,member=member,mean=float(z.prediction.mean()),bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e*e)))))
  else:
   _,_,tm,vm=next(z for z in w['efolds'] if z[0]=='DIAG10' and z[1]==fold);keys=set(w['elab'][tm][['farm','day']].itertuples(index=False,name=None))
   eo=w['eouter'];eo=eo[(eo.validator=='DIAG10')&(eo.seed==7)&(eo.farm==event.farm)&(eo.day==event.day)]
   for member in ['season_r3','season_pfn','season_v2']:
    e=eo[member]-eo.sub_ec;members.append(dict(target='EC',farm=event.farm,day=event.day,member=member,mean=float(eo[member].mean()),bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e*e)))))
  count=0
  for nk,nq in ds.items():
   if nk==key:continue
   if np.array_equal(q[weather].to_numpy(),nq[weather].to_numpy()):
    count+=1;cols=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog'];same=[c for c in cols if np.array_equal(q[c],nq[c],equal_nan=True)]
    overlaps.append(dict(target=event.target,farm=event.farm,day=event.day,other_farm=nk[0],other_day=nk[1],in_outer_train=nk in keys,same_internal_columns=';'.join(same),air_mean=float(nq.in_temp.mean()),temperature_mean=float(nq.sub_temp.mean()),temperature_gap=float(nq.sub_temp.mean()-nq.in_temp.mean()),temperature_rmse=float(np.sqrt(np.mean((nq.prediction-nq.sub_temp)**2))),air_curve_corr=float(np.corrcoef(q.in_temp,nq.in_temp)[0,1]),temperature_curve_corr=float(np.corrcoef(q.sub_temp,nq.sub_temp)[0,1])))
  summaries.append(dict(target=event.target,farm=event.farm,day=event.day,weather_twins=count))
 pd.DataFrame(overlaps).to_csv(H/'success_weather_overlaps.csv',index=False);pd.DataFrame(members).to_csv(H/'successful_member_scores.csv',index=False);pd.DataFrame(summaries).to_csv(H/'success_overlap_counts.csv',index=False)
 print(pd.DataFrame(overlaps).to_string(index=False));print(pd.DataFrame(members).to_string(index=False))
if __name__=='__main__':main()
