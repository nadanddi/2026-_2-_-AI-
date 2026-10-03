from pathlib import Path
import sys,json,hashlib,itertools,warnings
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; R=H.parents[3]
sys.path.insert(0,str(R/'집/클로드/research'))
import env
import numpy as np,pandas as pd
from scipy.stats import rankdata
warnings.filterwarnings('ignore',category=RuntimeWarning)
RAW=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','in_rad','act_vent','act_side','act_shade','act_thermal','act_valve','act_heating','act_circfan','act_co2','act_fog','act_cool','act_pump']
def mean(a):return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
def corr(a,b):
 m=np.isfinite(a)&np.isfinite(b)
 return float(np.corrcoef(a[m],b[m])[0,1]) if m.sum()>=4 and np.std(a[m])>0 and np.std(b[m])>0 else np.nan
def save(name,x):
 p=H/name;assert not p.exists(),p
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:float(a)),encoding='utf-8')
def load():
 raw=pd.read_csv(Path(env.DATA)/'train_X.csv',float_precision='round_trip');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].copy()
 raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[-2:].astype(int)
 t=pd.read_csv(R/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv',float_precision='round_trip')
 t=t[(t.validator=='DIAG10')&(t.base_seed==7)&(t.context=='1-8')].copy()
 e=pd.read_csv(R/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv',float_precision='round_trip');e=e[(e.validator=='DIAG10')&(e.seed==7)].copy()
 tr=t[t.member=='W30G'][['row_id','farm','day','hour','sub_temp','prediction']].rename(columns={'sub_temp':'truth','prediction':'prediction'})
 er=e[['row_id','farm','day','hour','sub_ec','season_v2']].rename(columns={'sub_ec':'truth','season_v2':'prediction'})
 return raw,t,e,{'TEMP':tr,'EC':er}
def inputs(raw):
 rows=[]
 pairs=[('in_temp','out_temp'),('in_temp','in_hum'),('in_temp','in_co2'),('in_hum','out_hum'),('act_vent','in_temp'),('act_vent','in_hum'),('act_vent','in_co2'),('act_shade','out_rad'),('act_thermal','out_rad'),('act_heating','in_temp'),('out_rad','in_temp'),('out_rad','in_co2'),('act_co2','in_co2'),('act_fog','in_hum'),('act_circfan','in_temp')]
 for (f,d),q in raw.groupby(['farm','day'],sort=True):
  q=q.sort_values('hour');z=dict(farm=f,day=int(d),late=float(d>=179));hour=q.hour.to_numpy()
  for c in RAW:
   a=q[c].to_numpy(float);v=a[np.isfinite(a)];dif=np.diff(a)
   z[c+'__missing']=float(np.isnan(a).mean());z[c+'__unique']=float(len(np.unique(v)))
   for k,value in [('mean',mean(a)),('std',float(np.std(v)) if len(v) else np.nan),('min',float(np.min(v)) if len(v) else np.nan),('max',float(np.max(v)) if len(v) else np.nan),('h0',float(a[0])),('night',mean(a[hour<=6])),('day',mean(a[(hour>=9)&(hour<=16)])),('evening',mean(a[hour>=17])),('zero_fraction',float(np.mean(v==0)) if len(v) else np.nan),('jump_max',float(np.nanmax(np.abs(dif))) if np.isfinite(dif).any() else np.nan),('jump_rms',float(np.sqrt(mean(dif*dif)))),('same_adjacent',float(np.mean(dif[np.isfinite(dif)]==0)) if np.isfinite(dif).any() else np.nan)]:z[c+'__'+k]=value
   z[c+'__daynight']=z[c+'__day']-z[c+'__night'];z[c+'__eveningnight']=z[c+'__evening']-z[c+'__night']
  for a,b in pairs:
   av=q[a].to_numpy(float);bv=q[b].to_numpy(float);z[a+'__corr__'+b]=corr(av,bv)
   for lag in [1,3]:z[a+f'__lead{lag}__'+b]=corr(av[:-lag],bv[lag:])
  z['air_outdoor_mean_gap']=z['in_temp__mean']-z['out_temp__mean']
  z['air_outdoor_std_ratio']=z['in_temp__std']/z['out_temp__std'] if z['out_temp__std']>0 else np.nan
  z['air_outdoor_jump_ratio']=z['in_temp__jump_rms']/z['out_temp__jump_rms'] if z['out_temp__jump_rms']>0 else np.nan
  rows.append(z)
 a=pd.DataFrame(rows).sort_values(['farm','day'])
 a['previous_record_gap']=a.day-a.groupby('farm').day.shift(1)
 for c in ['in_temp','in_hum','in_co2','out_temp','out_rad','act_vent','act_heating','act_shade','act_thermal']:
  for s in ['mean','h0']:
   v=c+'__'+s;prev=a.groupby('farm')[v].shift(1);a[v+'__previous']=prev;a[v+'__change_previous']=a[v]-prev
  a[c+'__h0_minus_previous_evening']=a[c+'__h0']-a.groupby('farm')[c+'__evening'].shift(1)
 # Same weather is only input equality, never a date/identity assertion.
 signatures={}
 for (f,d),q in raw.groupby(['farm','day']):
  ar=q.sort_values('hour')[['out_temp','out_hum','out_rad','out_wspd']].to_numpy();sig=ar.tobytes();signatures.setdefault(sig,[]).append((f,d))
 twin={k:len(v)-1 for v in signatures.values() for k in v};a['exact_weather_twins']=[twin[(f,d)] for f,d in zip(a.farm,a.day)]
 return a
def outcomes(raw,frames,t,e):
 allrows=[];hours=[]
 for target,p in frames.items():
  p=p.merge(raw[['row_id','in_temp']],on='row_id',validate='one_to_one');p['error']=p.prediction-p.truth;p['se']=p.error**2
  g=p.groupby(['farm','day']).agg(truth=('truth','mean'),prediction=('prediction','mean'),bias=('error','mean'),sse=('se','sum'),air=('in_temp','mean'),air_n=('in_temp','count'),n=('row_id','size'))
  g['rmse']=np.sqrt(g.sse/g.n);g['gap']=g.truth-g.air;g['level_sse_fraction']=g.n*g.bias**2/g.sse
  g['target']=target;g['event']=((g.gap.abs()>=2)&(g.air_n==24)) if target=='TEMP' else g.truth>=1
  cut=.5 if target=='TEMP' else .1;g['failure']=g.event&(g.rmse>cut);g['good']=g.event&~g.failure
  g['core_failure']=g.event&((g.bias*g.gap<0)&(g.bias.abs()>=.5) if target=='TEMP' else g.bias<=-.2)
  for label,threshold in [('strict1',.75 if target=='TEMP' else .15),('strict2',1 if target=='TEMP' else .2)]:g[label]=g.event&(g.rmse>threshold)
  if target=='TEMP':
   wide=t[t.member.isin(['BASE','CODEX','PFN'])].pivot(index='row_id',columns='member',values='prediction').reindex(p.row_id).to_numpy();names=['BASE','CODEX','PFN']
  else:wide=e.set_index('row_id').loc[p.row_id,['season_r3','season_pfn']].to_numpy();names=['R3','PFN']
  pp=p.copy();pp['all_under']=(wide<p.truth.to_numpy()[:,None]).all(1);pp['all_over']=(wide>p.truth.to_numpy()[:,None]).all(1);pp['member_disagreement']=np.std(wide,axis=1)
  ag=pp.groupby(['farm','day']).agg(all_under_fraction=('all_under','mean'),all_over_fraction=('all_over','mean'),member_disagreement=('member_disagreement','mean'));g=g.join(ag)
  for i,name in enumerate(names):
   pp[name+'_error']=wide[:,i]-p.truth.to_numpy();g[name+'_bias']=pp.groupby(['farm','day'])[name+'_error'].mean()
  for phase,mask in [('night',p.hour<=6),('day',p.hour.between(7,16)),('evening',p.hour>=17)]:
   h=p[mask].groupby(['farm','day']).agg(phase_bias=('error','mean'),phase_sse=('se','sum'));g[phase+'_bias']=h.phase_bias;g[phase+'_sse_fraction']=h.phase_sse/g.sse
  hours.append(pp.assign(target=target));allrows.append(g.reset_index())
 return pd.concat(allrows,ignore_index=True),pd.concat(hours,ignore_index=True)
def bh(p):
 ix=np.argsort(p);q=np.empty(len(p));q[ix]=np.minimum(1,np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q
def scan(d,cols):
 rows=[];rng=np.random.default_rng(20261003);B=1999
 for target,q in d.groupby('target',sort=True):
  q=q.reset_index(drop=True);f=q.failure.to_numpy();strata=q.farm+'_'+q.late.astype(int).astype(str);conditions=[];meta=[]
  for c in cols:
   v=q[c].to_numpy(float);valid=np.isfinite(v)
   if valid.sum()<10 or np.unique(v[valid]).size<2:continue
   for quant in [.25,.5,.75]:
    cut=float(np.quantile(v[valid],quant))
    for op in ['>=','<=']:
     a=(v>=cut) if op=='>=' else (v<=cut);a=a&valid
     if a[valid].all() or (~a[valid]).all():continue
     conditions.append(a.astype(float));meta.append((c,op,cut,quant,valid))
  A=np.array(conditions).T;V=np.array([z[4] for z in meta]).T.astype(float)
  # Compare observations against a random failure subset within identical farm/period strata.
  P=np.empty((len(q),B),dtype=np.float32)
  for s in sorted(strata.unique()):
   ix=np.where(strata==s)[0];m=f[ix].astype(np.float32)
   P[ix,:]=np.column_stack([rng.permutation(m) for _ in range(B)])
  with np.errstate(divide='ignore',invalid='ignore'):
   sims=(A.T@P)/(V.T@P);observed=(A.T@f)/(V.T@f);center=np.nanmean(sims,axis=1)
  for k,(c,op,cut,quant,valid) in enumerate(meta):
   a=A[:,k].astype(bool);ff=f&valid;nf=~f&valid;ev=q.event.to_numpy()&~f&valid;weights=[];bg=[];stratdetail=[]
   for s in sorted(strata.unique()):
    sf=(strata==s).to_numpy()&ff;sn=(strata==s).to_numpy()&nf
    if sf.any() and sn.any():weights.append(sf.sum());bg.append(a[sn].mean());stratdetail.append(dict(stratum=s,failure_n=int(sf.sum()),failure_yes=int(a[sf].sum()),other_n=int(sn.sum()),other_yes=int(a[sn].sum())))
   matched=float(np.average(bg,weights=weights)) if weights else np.nan
   vals=sims[k];vals=vals[np.isfinite(vals)];p=(1+np.sum(np.abs(vals-center[k])>=abs(observed[k]-center[k])-1e-12))/(len(vals)+1)
   row=dict(target=target,feature=c,operator=op,cut=cut,quantile=quant,failure_n=int(ff.sum()),failure_yes=int(a[ff].sum()),failure_rate=float(a[ff].mean()) if ff.any() else np.nan,other_n=int(nf.sum()),other_rate=float(a[nf].mean()) if nf.any() else np.nan,event_nonfailure_n=int(ev.sum()),event_nonfailure_rate=float(a[ev].mean()) if ev.any() else np.nan,matched_other_rate=matched,lift_matched=float(observed[k]-matched),p_permutation=float(p),strata=json.dumps(stratdetail,ensure_ascii=False),leave_one_failure_min=float((a[ff].sum()-int(a[ff].any()))/(ff.sum()-1)) if ff.sum()>1 else np.nan)
   for lab in ['core_failure','strict1','strict2']:
    ss=q[lab].to_numpy()&valid;row[lab+'_n']=int(ss.sum());row[lab+'_rate']=float(a[ss].mean()) if ss.any() else np.nan
   rows.append(row)
  print(target,'days',len(q),'failure',f.sum(),'features',len(cols),'conditions',len(meta),flush=True)
 out=pd.DataFrame(rows);out['q_BH']=bh(out.p_permutation.to_numpy());out['p_Bonferroni']=np.minimum(1,out.p_permutation*len(out));return out
def main():
 raw,t,e,frames=load();a=inputs(raw);g,h=outcomes(raw,frames,t,e);d=g.merge(a,on=['farm','day'],validate='many_to_one');cols=list(a.columns[2:])
 a.to_csv(H/'all_input_features.csv',index=False);d.to_csv(H/'all_daily_cases.csv',index=False);h.to_csv(H/'public_hourly_cases.csv',index=False)
 ss=scan(d,cols);ss.to_csv(H/'all_conditions.csv',index=False)
 majority=ss[(ss.failure_rate>=.5)&(ss.failure_n>=5)].sort_values(['target','lift_matched','failure_rate'],ascending=[True,False,False]);majority.to_csv(H/'common_conditions.csv',index=False)
 summary=[]
 for target,q in d.groupby('target'):
  for label in ['all','event','failure','core_failure','strict1','strict2','good']:
   z=q if label=='all' else q[q[label]];summary.append(dict(target=target,group=label,n=len(z),sse_fraction=float(z.sse.sum()/q.sse.sum()),rmse=float(np.sqrt(z.sse.sum()/z.n.sum())) if len(z) else None,mean_bias=float(z.bias.mean()) if len(z) else None,mean_level_fraction=float(z.level_sse_fraction.mean()) if len(z) else None,F13=int((z.farm=='F13').sum()),F47=int((z.farm=='F47').sum()),late=int(z.late.sum()),warm=int((z.gap>=2).sum()) if target=='TEMP' else None))
 pd.DataFrame(summary).to_csv(H/'cohort_summary.csv',index=False)
 save('manifest.json',dict(status='EXPLORATORY_NOT_CAUSAL',features=len(cols),conditions=len(ss),permutations=1999,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),raw_input_rows=len(raw),temp_days=int((d.target=='TEMP').sum()),ec_days=int((d.target=='EC').sum()),reserved_EC_labels_read=False,new_models=False))
 print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
