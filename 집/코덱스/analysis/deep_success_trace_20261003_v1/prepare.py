from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import pandas as pd,numpy as np,joblib
O=R/'집/코덱스/local/deep_success_trace_20261003_v1';O.mkdir(parents=True,exist_ok=True)
TEMP=[('F13',51),('F13',143),('F13',194),('F47',72),('F47',191),('F13',57),('F47',68)]
EC=[('F13',177),('F47',139),('F13',135),('F47',138)]
def savej(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda a:float(a)),encoding='utf-8')
def main():
 lab,ct,phc,w,folds,outer=S.loadtemp();elab,core,wv,efolds,eouter=S.loadec()
 selected=[]
 for target,data,fs,cases in [('TEMP',lab,folds,TEMP),('EC',elab,efolds,EC)]:
  for f,day in cases:
   if target=='TEMP':matches=[(k,fd) for name,k,fd in fs if name=='DIAG10' and day in fd[f]];k,fd=matches[0];tm,vm=S.common.split_mask(data,fd)
   else:matches=[(k,tm,vm) for name,k,tm,vm in fs if name=='DIAG10' and data.loc[vm].eval('farm == @f and day == @day').any()];k,tm,vm=matches[0]
   assert len(matches)==1;selected.append(dict(target=target,farm=f,day=day,fold=k))
 events=pd.read_csv(R/'집/코덱스/analysis/success_signals_20261003_v1/event_days.csv')
 joblib.dump(dict(lab=lab,ct=ct,phc=phc,w=w,folds=folds,outer=outer,elab=elab,wv=wv,efolds=efolds,eouter=eouter,selected=selected),O/'world.joblib')
 pd.DataFrame(selected).to_csv(H/'selected_cases.csv',index=False)
 pd.DataFrame({'column':sorted(set(ct+phc+S.FEATURE_COLUMNS))}).to_csv(H/'temperature_columns.csv',index=False)
 print('PREPARED',selected,flush=True)
 # Raw data quality and alignment diagnostics: temperature only.
 raw=pd.read_csv(Path(S.env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].copy();raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[-2:].astype(int)
 yy=pd.read_csv(Path(S.env.DATA)/'train_y.csv',usecols=['row_id','sub_temp']);raw=raw.merge(yy,on='row_id');allp=outer[(outer.validator=='DIAG10')&(outer.base_seed==7)&(outer.context=='1-8')&(outer.member=='W30G')][['row_id','prediction']];raw=raw.merge(allp,on='row_id')
 rows=[];alignment=[];neighbors=[];coverage=[]
 for f,day in TEMP:
  q=raw[(raw.farm==f)&(raw.day==day)].sort_values('hour');trmeta=next(z for z in selected if z['target']=='TEMP' and z['farm']==f and z['day']==day);fd=next(fd for n,k,fd in folds if n=='DIAG10' and k==trmeta['fold']);tm,vm=S.common.split_mask(lab,fd)
  for col in ['in_temp','in_hum','in_co2','sub_temp']:
   arr=q[col].to_numpy(float);dif=np.diff(arr);rows.append(dict(farm=f,day=day,column=col,missing=int(np.isnan(arr).sum()),n_unique=int(q[col].nunique()),same_adjacent=int((dif==0).sum()),max_jump=float(np.nanmax(np.abs(dif))),median_second_diff=float(np.nanmedian(np.abs(np.diff(arr,2))))))
  e=q.prediction.to_numpy()-q.sub_temp.to_numpy();level=float(e.mean())
  for lag in range(-6,7):
   ix=np.arange(24);good=(ix+lag>=0)&(ix+lag<24);pred=q.prediction.to_numpy()[ix[good]];y=q.sub_temp.to_numpy()[(ix+lag)[good]];err=pred-y;alignment.append(dict(farm=f,day=day,lag=lag,n=int(good.sum()),rmse=float(np.sqrt(np.mean(err*err))),bias=float(err.mean()),centered_rmse=float(np.sqrt(np.mean((err-err.mean())**2)))))
  tr=lab[tm].copy();cols=list(S.FEATURE_COLUMNS);med=tr[cols].median();filled=tr[cols].fillna(med).fillna(0);sd=filled.std(ddof=0).replace(0,1);dv=lab[(lab.farm==f)&(lab.day==day)].sort_values('hour')
  for nf,nd in tr[(tr.farm==f)&((tr.day>=179)==(day>=179))].groupby(['farm','day']).groups:
   nn=tr[(tr.farm==nf)&(tr.day==nd)].sort_values('hour');dist=float(np.sqrt(np.mean(((nn[cols].fillna(med).fillna(0).to_numpy()-dv[cols].fillna(med).fillna(0).to_numpy())/sd.to_numpy())**2)))
   nr=raw[(raw.farm==nf)&(raw.day==nd)];neighbors.append(dict(farm=f,day=day,neighbor_farm=nf,neighbor_day=nd,distance=dist,truth=float(nr.sub_temp.mean()),air=float(nr.in_temp.mean()),gap=float(nr.sub_temp.mean()-nr.in_temp.mean()),weight=float(w[tm][tr.row_id.isin(nn.row_id)].mean())))
  ids=set(tr.row_id);trr=raw[raw.row_id.isin(ids)];daily=trr.groupby(['farm','day']).agg(y=('sub_temp','mean'),air=('in_temp','mean'),n=('in_temp','count'));daily['gap']=daily.y-daily.air
  for samefarm in [False,True]:
   dq=daily[(daily.n==24)&((daily.index.get_level_values(1)>=179)==(day>=179))]
   if samefarm:dq=dq[dq.index.get_level_values(0)==f]
   coverage.append(dict(farm=f,day=day,same_farm=samefarm,n=len(dq),warm=int((dq.gap>=2).sum()),negative=int((dq.gap<0).sum()),max_gap=float(dq.gap.max())))
  # Weights are aligned to the actual world and derived only from original inputs.
  ws=w[lab.row_id.isin(q.row_id)];rows.append(dict(farm=f,day=day,column='training_weight_if_fitted',missing=0,n_unique=len(np.unique(ws)),same_adjacent=None,max_jump=None,median_second_diff=None,weight_mean=float(ws.mean())))
 pd.DataFrame(rows).to_csv(H/'sensor_quality.csv',index=False);pd.DataFrame(alignment).to_csv(H/'alignment.csv',index=False)
 nn=pd.DataFrame(neighbors).sort_values(['farm','day','distance']);nn['rank']=nn.groupby(['farm','day']).cumcount()+1;nn.to_csv(H/'actual_feature_neighbors.csv',index=False)
 pd.DataFrame(coverage).to_csv(H/'train_coverage.csv',index=False)
 raw.to_csv(O/'temperature_raw_public.csv',index=False)
 # All severe-event feature relations, including controls not chosen for model work.
 feats=[]
 for f,day in events[['farm','day']].drop_duplicates().itertuples(index=False,name=None):
  q=raw[(raw.farm==f)&(raw.day==day)].sort_values('hour');d=dict(farm=f,day=int(day))
  for col in S.common.USABLE:
   if col not in q:continue
   v=q[col];d[col+'_mean']=float(v.mean());d[col+'_std']=float(v.std(ddof=0));d[col+'_h0']=float(v.iloc[0]);d[col+'_diff_rms']=float(np.sqrt(np.nanmean(np.diff(v.to_numpy())**2)))
  for c in ['in_temp','in_hum','in_co2','out_temp']:
   d[c+'_daynight']=float(q[q.hour.between(9,16)][c].mean()-q[q.hour<=6][c].mean())
  d['air_outdoor_correlation']=float(q.in_temp.corr(q.out_temp));d['air_outdoor_std_ratio']=float(q.in_temp.std(ddof=0)/q.out_temp.std(ddof=0));d['air_humidity_correlation']=float(q.in_temp.corr(q.in_hum));d['air_CO2_correlation']=float(q.in_temp.corr(q.in_co2));d['vent_air_correlation']=float(q.act_vent.corr(q.in_temp));d['shade_rad_correlation']=float(q.act_shade.corr(q.out_rad));d['thermal_rad_correlation']=float(q.act_thermal.corr(q.out_rad));d['air_outdoor_mean_gap']=float((q.in_temp-q.out_temp).mean());feats.append(d)
 feature=pd.DataFrame(feats);ed=events[['target','farm','day','good','relaxed_good','gap','rmse']].merge(feature,on=['farm','day']);ed.to_csv(H/'event_relations.csv',index=False)
 contrasts=[]
 for target,q in ed.groupby('target'):
  for col in feature.columns[2:]:
   pos=q[q.good][col];neg=q[~q.good][col];valid=pos.notna().all() and neg.notna().sum()>0
   contrasts.append(dict(target=target,feature=col,good_n=len(pos),bad_n=len(neg),good_mean=float(pos.mean()),bad_mean=float(neg.mean()),difference=float(pos.mean()-neg.mean()),all_good_above_bad_median=bool(valid and (pos>=neg.median()).all()),all_good_below_bad_median=bool(valid and (pos<=neg.median()).all()),relaxed_difference=float(q[q.relaxed_good][col].mean()-q[~q.relaxed_good][col].mean())))
 pd.DataFrame(contrasts).to_csv(H/'relation_contrasts.csv',index=False)
 savej(H/'preparation.json',dict(status='PASS',selected=selected,temperature_columns=len(set(ct+phc+S.FEATURE_COLUMNS)),relations=len(feature.columns)-2,world_sha256=hashlib.sha256((O/'world.joblib').read_bytes()).hexdigest()))
 print('DIAGNOSTICS_DONE',flush=True)
if __name__=='__main__':main()
