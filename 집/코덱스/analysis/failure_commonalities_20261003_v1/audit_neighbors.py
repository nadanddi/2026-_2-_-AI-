from analyze import *
sys.path.insert(0,str(R/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S,joblib
from scipy.spatial.distance import cdist
def main():
 w=joblib.load(R/'집/코덱스/local/deep_success_trace_20261003_v1/world.joblib');c=pd.read_csv(H/'actual_training_coverage_v2.csv',float_precision='round_trip');n=pd.read_csv(H/'actual_training_neighbors_v2.csv',float_precision='round_trip');d=pd.read_csv(H/'final_daily_cases_v2.csv',float_precision='round_trip').set_index(['target','farm','day']);checks=0;mx=0.
 for z in c.itertuples():
  if z.target=='TEMP':
   data=w['lab'];fd=next(fd for name,k,fd in w['folds'] if name=='DIAG10' and k==z.fold);trainkeys={(f,int(day)) for f,day in data[['farm','day']].itertuples(index=False,name=None) if int(day) not in fd[f]};tr=data[[v in trainkeys for v in data[['farm','day']].itertuples(index=False,name=None)]];va=data[(data.farm==z.farm)&(data.day==z.day)];cols=list(S.FEATURE_COLUMNS)
  else:
   data=w['elab'];_,_,tm,vm=next(ff for ff in w['efolds'] if ff[0]=='DIAG10' and ff[1]==z.fold);tr,va=S.seasonal(data[tm],data[vm],w['wv']);va=va[(va.farm==z.farm)&(va.day==z.day)];trainkeys=set(tr[['farm','day']].itertuples(index=False,name=None));cols=[v for v in S.loadcore().FULL if v!='day']+['season']
  assert (z.farm,z.day) not in trainkeys;assert len(trainkeys)==z.train_days;checks+=2
  subset={k for k in trainkeys if k[0]==z.farm and (k[1]>=179)==(z.day>=179)}
  rawdaily=d.loc[z.target];sg=rawdaily.loc[rawdaily.index.isin(subset)]
  if z.target=='TEMP':sg=sg[sg.air_n==24];assert int((sg.gap>=2).sum())==z.samefarm_warm;assert int((sg.gap<=-2).sum())==z.samefarm_cool;checks+=2
  else:assert int((sg.truth>=1).sum())==z.samefarm_high;checks+=1
  assert len(sg)==z.samefarm_period_days;checks+=1
  saved=n[(n.target==z.target)&(n.farm==z.farm)&(n.day==z.day)]
  assert all((rr.neighbor_farm,rr.neighbor_day) in trainkeys for rr in saved.itertuples());checks+=len(saved)
  # A separate Euclidean distance implementation for representative failures and successes.
  if (z.target,z.farm,z.day) in [('TEMP','F47',191),('TEMP','F13',51),('TEMP','F47',158),('EC','F47',231),('EC','F47',139),('EC','F13',177)]:
   med=tr[cols].median();std=tr[cols].fillna(med).fillna(0).std(ddof=0).replace(0,1);q=((va.sort_values('hour')[cols].fillna(med).fillna(0))/std).to_numpy().reshape(1,-1)
   for rr in saved.itertuples():
    a=tr[(tr.farm==rr.neighbor_farm)&(tr.day==rr.neighbor_day)].sort_values('hour');ar=(a[cols].fillna(med).fillna(0)/std).to_numpy().reshape(1,-1);dist=float(cdist(q,ar)[0,0]/math.sqrt(ar.size))
    diff=abs(dist-rr.distance);mx=max(mx,diff);assert diff<1e-10;checks+=1
 save('training_split_audit.json',dict(status='PASS',checks=checks,all_events=51,independent_distance_cases=6,max_distance_difference=mx,EC_reserved_access=False))
 print('PASS',checks,mx)
if __name__=='__main__':
 import math
 main()
