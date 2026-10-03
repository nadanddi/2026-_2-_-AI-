from analyze import *
RULES={
 '환기 하루평균≤10':('act_vent__mean','<=',10),
 '환기0인시간≥18시간':('act_vent__zero_fraction','>=',.75),
 '환기0인시간≥20시간':('act_vent__zero_fraction','>=',20/24),
 '밤환기평균=0':('act_vent__night','<=',0),
 '외기하루평균≤10℃':('out_temp__mean','<=',10),
 '차광 낮−밤≥90':('act_shade__daynight','>=',90),
 '보온 낮−밤≥90':('act_thermal__daynight','>=',90),
 '낮CO2가밤보다높음':('in_co2__daynight','>',0),
 '실내 낮−밤온도≥5℃':('in_temp__daynight','>=',5),
 '난방하루평균=0':('act_heating__mean','<=',0),
 '순환팬저녁평균=0':('act_circfan__evening','<=',0),
 'CO2구동0인시간≤21시간':('act_co2__zero_fraction','<=',21/24),
 '0시실내습도≤76':('in_hum__h0','<=',76),
 '직전관측일일사평균≥114.833':('out_rad__mean__previous','>=',114.83333333333333),
 '실내외온도상관≤.905626':('in_temp__corr__out_temp','<=',.905626),
 'CO2시간최대점프≥33':('in_co2__jump_max','>=',33),
 '직전관측일간격=1':('previous_record_gap','<=',1),
 '같은외기전체값기록있음':('exact_weather_twins','>=',1),
 '하루수준오차비중≥50%':('level_sse_fraction','>=',.5),
 '하루수준오차비중≥75%':('level_sse_fraction','>=',.75),
 '모든전문가과소예측≥18시간':('all_under_fraction','>=',.75),
 '모든전문가과대예측≥18시간':('all_over_fraction','>=',.75),
}
def condition(v,op,cut):return {'<=':v<=cut,'>=':v>=cut,'>':v>cut}[op]&np.isfinite(v)
def main():
 d=pd.read_csv(H/'final_daily_cases_v2.csv',float_precision='round_trip');out=[]
 for target,q in d.groupby('target'):
  for group in ['all','failure','core_failure','strict2','good']:
   z=q if group=='all' else q[q[group]]
   for name,(c,op,cut) in RULES.items():
    v=z[c];valid=v.notna();a=condition(v,op,cut)
    out.append(dict(target=target,group=group,rule=name,feature=c,operator=op,cut=cut,n=len(z),available=int(valid.sum()),yes=int(a.sum()),fraction=float(a[valid].mean()) if valid.any() else None))
 pd.DataFrame(out).to_csv(H/'fixed_rule_counts_v3.csv',index=False)
 raw,_,_,_=load();quality=[]
 for c in RAW:
  for target,q in d.groupby('target'):
   ids=set(q[['farm','day']].itertuples(index=False,name=None));z=raw[[tuple(v) in ids for v in raw[['farm','day']].to_numpy()]];v=z[c];quality.append(dict(target=target,column=c,total=len(z),missing=int(v.isna().sum()),unique=int(v.nunique()),minimum=float(v.min()),maximum=float(v.max())))
 pd.DataFrame(quality).to_csv(H/'raw_quality_v3.csv',index=False)
 a=pd.read_csv(H/'actual_training_coverage_v2.csv',float_precision='round_trip').merge(d[['target','farm','day','failure','core_failure','late']],on=['target','farm','day']);summ=[]
 for tar,q in a.groupby('target'):
  for group in ['failure','core_failure','good']:
   z=q[q[group] if group!='good' else q.good];r=dict(target=tar,group=group,n=len(z),nearest_distance_median=float(z.nearest_distance.median()))
   if tar=='TEMP':r.update(top5_no_severe_same_sign=int((z.top5_severe_same_sign==0).sum()),samefarm_period_no_warm=int(((z.query_gap>=2)&(z.samefarm_warm==0)).sum()),beyond_samefarm_gap_range=int(((z.query_gap<z.train_gap_min)|(z.query_gap>z.train_gap_max)).sum()))
   else:r.update(top5_no_high=int((z.top5_high==0).sum()),samefarm_period_no_high=int((z.samefarm_high==0).sum()),beyond_samefarm_truth_range=int((z.query_truth>z.train_truth_max).sum()),mean_top5_truth_gap=float(z.top5_truth_gap.mean()))
   summ.append(r)
 pd.DataFrame(summ).to_csv(H/'coverage_summary_v3.csv',index=False)
 print(pd.DataFrame(out).query("group in ['failure','core_failure','good']")[['target','group','rule','yes','available']].to_string(index=False));print(pd.DataFrame(summ).to_string(index=False))
if __name__=='__main__':main()
