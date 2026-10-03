from prepare import *
def main():
 raw=pd.read_csv(O/'temperature_raw_public.csv',float_precision='round_trip');events=pd.read_csv(R/'집/코덱스/analysis/success_signals_20261003_v1/event_days.csv');rows=[];pair=[];metadata=[];world=joblib.load(O/'world.joblib')
 for f,day in events[events.target=='TEMP'][['farm','day']].itertuples(index=False,name=None):
  q=raw[(raw.farm==f)&(raw.day==day)].sort_values('hour');ev=events[(events.target=='TEMP')&(events.farm==f)&(events.day==day)].iloc[0]
  for label,mask in [('night0_6',q.hour<=6),('day7_16',q.hour.between(7,16)),('evening17_23',q.hour>=17)]:
   qq=q[mask];rows.append(dict(farm=f,day=int(day),good=bool(ev.good),segment=label,n=len(qq),gap=float((qq.sub_temp-qq.in_temp).mean()),bias=float((qq.prediction-qq.sub_temp).mean()),air=float(qq.in_temp.mean()),truth=float(qq.sub_temp.mean())))
 for fa,da,fb,db in [('F47',190,'F47',191),('F13',194,'F47',191),('F13',56,'F47',191)]:
  a=raw[(raw.farm==fa)&(raw.day==da)].sort_values('hour');b=raw[(raw.farm==fb)&(raw.day==db)].sort_values('hour');pair.append(dict(first=f'{fa}/{da}',second=f'{fb}/{db}',same_weather_values=int(np.sum(a[['out_temp','out_hum','out_rad','out_wspd']].to_numpy()==b[['out_temp','out_hum','out_rad','out_wspd']].to_numpy())),air_difference=float(b.in_temp.mean()-a.in_temp.mean()),truth_difference=float(b.sub_temp.mean()-a.sub_temp.mean()),prediction_difference=float(b.prediction.mean()-a.prediction.mean()),target_centered_curve_rmse=float(np.sqrt(np.mean(((b.sub_temp.to_numpy()-b.sub_temp.mean())-(a.sub_temp.to_numpy()-a.sub_temp.mean()))**2))),target_curve_correlation=float(np.corrcoef(a.sub_temp,b.sub_temp)[0,1])))
 for target,f,day in events[events.good][['target','farm','day']].itertuples(index=False,name=None):
  q=raw[(raw.farm==f)&(raw.day==day)].sort_values('hour');metadata.append(dict(target=target,farm=f,day=int(day),in_temp_h0=float(q.iloc[0].in_temp),in_co2_h0=float(q.iloc[0].in_co2),in_temp_daynight=float(q[q.hour.between(9,16)].in_temp.mean()-q[q.hour<=6].in_temp.mean()),outside_temp_std=float(q.out_temp.std(ddof=0)),inside_temp_std=float(q.in_temp.std(ddof=0))))
 pd.DataFrame(rows).to_csv(H/'event_temperature_phases.csv',index=False);pd.DataFrame(pair).to_csv(H/'same_weather_pair_offsets.csv',index=False);pd.DataFrame(metadata).to_csv(H/'successful_raw_summary.csv',index=False)
 pd.DataFrame(pair).to_string(index=False);print(pd.DataFrame(pair).to_string(index=False));print(pd.DataFrame(rows).query('good or day == 191').to_string(index=False));print(pd.DataFrame(metadata).to_string(index=False))
if __name__=='__main__':main()
