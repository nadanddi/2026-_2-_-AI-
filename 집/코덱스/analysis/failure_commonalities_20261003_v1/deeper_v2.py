from analyze import *
sys.path.insert(0,str(R/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import joblib
def main():
 d=pd.read_csv(H/'all_daily_cases.csv',float_precision='round_trip');d.loc[d.target=='EC','gap']=np.nan
 d.to_csv(H/'final_daily_cases_v2.csv',index=False)
 ss=pd.read_csv(H/'all_conditions.csv',float_precision='round_trip');rng=np.random.default_rng(20261004);extra=[]
 for target,q in d.groupby('target'):
  q=q.reset_index(drop=True);f=q.failure.to_numpy();rows=ss[ss.target==target];strata=q.farm+'_'+(q.day//30).astype(str)
  P=np.empty((len(q),1999),dtype=np.float32)
  for s in sorted(strata.unique()):
   ix=np.where(strata==s)[0];P[ix]=np.column_stack([rng.permutation(f[ix].astype(np.float32)) for _ in range(1999)])
  A=[];V=[]
  for r in rows.itertuples():
   v=q[r.feature].to_numpy();valid=np.isfinite(v);a=v>=r.cut if r.operator=='>=' else v<=r.cut;A.append((a&valid).astype(float));V.append(valid.astype(float))
  A=np.array(A);V=np.array(V)
  with np.errstate(divide='ignore',invalid='ignore'):sims=(A@P)/(V@P);obs=(A@f)/(V@f)
  for i,r in enumerate(rows.itertuples()):
   sim=sims[i];sim=sim[np.isfinite(sim)];center=mean(sim);p=(1+np.sum(np.abs(sim-center)>=abs(obs[i]-center)-1e-12))/(len(sim)+1)
   extra.append(dict(index=r.Index,p_fine_date=p))
 b=pd.DataFrame(extra).set_index('index');ss['p_fine_date']=b.loc[ss.index,'p_fine_date'];ss['q_fine_date_BH']=bh(ss.p_fine_date.to_numpy());ss['fine_date_bonf']=np.minimum(1,ss.p_fine_date*len(ss));ss.to_csv(H/'conditions_with_date_control_v2.csv',index=False)
 world=joblib.load(R/'집/코덱스/local/deep_success_trace_20261003_v1/world.joblib');records=[];nnrows=[]
 for target,data,folds in [('TEMP',world['lab'],world['folds']),('EC',world['elab'],world['efolds'])]:
  daily=d[d.target==target].set_index(['farm','day']);chosen=d[(d.target==target)&d.event]
  for f,day in chosen[['farm','day']].itertuples(index=False,name=None):
   if target=='TEMP':
    k,fd=next((k,fd) for n,k,fd in folds if n=='DIAG10' and day in fd[f]);tm,vm=S.common.split_mask(data,fd);cols=list(S.FEATURE_COLUMNS);tr=data[tm];va=data[vm]
   else:
    k,tm,vm=next((k,tm,vm) for n,k,tm,vm in folds if n=='DIAG10' and ((data.loc[vm,'farm']==f)&(data.loc[vm,'day']==day)).any());tr,va=S.seasonal(data[tm],data[vm],world['wv']);cols=[c for c in S.loadcore().FULL if c!='day']+['season']
   ids=set(tr.row_id);trdays=set(tr[['farm','day']].itertuples(index=False,name=None));assert (f,day) not in trdays
   td=daily.loc[daily.index.isin(trdays)].copy();same=td[(td.index.get_level_values(0)==f)&((td.index.get_level_values(1)>=179)==(day>=179))]
   z=dict(target=target,farm=f,day=int(day),fold=k,good=bool(daily.loc[(f,day),'good']),train_days=len(td),samefarm_period_days=len(same))
   if target=='TEMP':
    same=same[same.air_n==24];z.update(samefarm_period_days=len(same),samefarm_warm=int((same.gap>=2).sum()),samefarm_cool=int((same.gap<=-2).sum()),train_gap_min=float(same.gap.min()),train_gap_max=float(same.gap.max()),query_gap=float(daily.loc[(f,day),'gap']))
   else:z.update(samefarm_high=int((same.truth>=1).sum()),train_truth_min=float(same.truth.min()),train_truth_max=float(same.truth.max()),query_truth=float(daily.loc[(f,day),'truth']))
   med=tr[cols].median();sd=tr[cols].fillna(med).fillna(0).std(ddof=0).replace(0,1);dv=va[(va.farm==f)&(va.day==day)].sort_values('hour')[cols].fillna(med).fillna(0).to_numpy()
   dist=[]
   for (nf,nd),nq in tr.groupby(['farm','day']):
    if nf!=f or (nd>=179)!=(day>=179):continue
    arr=nq.sort_values('hour')[cols].fillna(med).fillna(0).to_numpy();distance=float(np.sqrt(np.mean(((arr-dv)/sd.to_numpy())**2)));dr=daily.loc[(nf,nd)]
    dist.append(dict(target=target,farm=f,day=int(day),neighbor_farm=nf,neighbor_day=int(nd),distance=distance,neighbor_truth=float(dr.truth),neighbor_gap=float(dr.gap),neighbor_own_oof_rmse=float(dr.rmse)))
   dist=sorted(dist,key=lambda z:z['distance'])
   for rank,nr in enumerate(dist,1):nr['rank']=rank;nnrows.append(nr)
   top=dist[:5];z['nearest_distance']=top[0]['distance'] if top else np.nan
   if target=='TEMP':z['top5_relation_same_sign']=sum(np.sign(v['neighbor_gap'])==np.sign(daily.loc[(f,day),'gap']) for v in top);z['top5_severe_same_sign']=sum(abs(v['neighbor_gap'])>=2 and np.sign(v['neighbor_gap'])==np.sign(daily.loc[(f,day),'gap']) for v in top)
   else:z['top5_high']=sum(v['neighbor_truth']>=1 for v in top);z['top5_truth_mean']=mean(np.array([v['neighbor_truth'] for v in top]));z['top5_truth_gap']=float(daily.loc[(f,day),'truth'])-z['top5_truth_mean']
   records.append(z)
  print('COVERAGE',target,len(records),flush=True)
 pd.DataFrame(records).to_csv(H/'actual_training_coverage_v2.csv',index=False);pd.DataFrame(nnrows).to_csv(H/'actual_training_neighbors_v2.csv',index=False)
 print(pd.DataFrame(records).to_string(index=False),flush=True)
if __name__=='__main__':main()
