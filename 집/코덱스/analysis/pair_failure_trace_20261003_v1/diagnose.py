from pathlib import Path
import sys,json,math
import pandas as pd
import numpy as np
H=Path(__file__).resolve().parent; R=H.parents[3]
sys.path.insert(0,str(R/'집/클로드/research'));import env
RAW=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog','out_temp','out_hum','out_rad','out_wspd']
W=['out_temp','out_hum','out_rad','out_wspd']; CASES=[('F13',194),('F47',191)]
def main():
 x=pd.read_csv(R/'공용/대회자료/정형데이터/참가자_배포/train_X.csv',usecols=['row_id']+RAW)
 y=pd.read_csv(R/'공용/대회자료/정형데이터/참가자_배포/train_y.csv',usecols=['row_id','sub_temp'])
 x=x.merge(y,on='row_id',validate='one_to_one');x['farm']=x.row_id.str[:3];x['day']=x.row_id.str[4:7].astype(int);x['hour']=x.row_id.str[8:10].astype(int)
 x=x[x.farm.isin(['F13','F47'])].sort_values(['farm','day','hour'])
 t=pd.read_csv(R/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv');t=t[t.validator=='DIAG10']
 p=t[(t.base_seed==7)&(t.context=='1-8')].pivot(index='row_id',columns='member',values='prediction')
 a=x.join(p,on='row_id');a['gap']=a.sub_temp-a.in_temp;a['error']=a.W30G-a.sub_temp
 d=a.groupby(['farm','day']).agg(y=('sub_temp','mean'),air=('in_temp','mean'),pred=('W30G','mean'),gap=('gap','mean'),bias=('error','mean'),rmse=('error',lambda z:np.sqrt(np.mean(z*z))),air_n=('in_temp','count'))
 feats=x.groupby(['farm','day'])[RAW].mean()
 stats=[];hours=[];neighbors=[];weather=[];timeline=[];robust=[]
 for f,day in CASES:
  q=a[(a.farm==f)&(a.day==day)].sort_values('hour').copy();e=q.error.to_numpy();bias=float(e.mean());sse=float(e@e)
  members=q[['BASE','CODEX','PFN']].to_numpy();lo=members.min(axis=1);hi=members.max(axis=1);oracle=np.clip(q.sub_temp.to_numpy(),lo,hi);oe=oracle-q.sub_temp.to_numpy()
  stats.append(dict(farm=f,day=day,air=float(q.in_temp.mean()),truth=float(q.sub_temp.mean()),pred=float(q.W30G.mean()),gap=float(q.gap.mean()),pred_gap=float((q.W30G-q.in_temp).mean()),bias=bias,rmse=float(np.sqrt(sse/24)),level_fraction=24*bias*bias/sse,centered_rmse=float(np.sqrt(np.mean((e-bias)**2))),under_hours=int((e<0).sum()),above_experts_hours=int((q.sub_temp.to_numpy()>hi).sum()),oracle_rmse=float(np.sqrt(np.mean(oe**2))),oracle_remaining_sse=float((oe@oe)/sse),expert_means={c:float(q[c].mean()) for c in ['BASE','CODEX','PFN']}))
  for label,mask in [('밤0–6',q.hour<=6),('낮7–16',q.hour.between(7,16)),('저녁17–23',q.hour>=17)]:
   z=q[mask];stats.append(dict(farm=f,day=day,segment=label,n=len(z),air=float(z.in_temp.mean()),truth=float(z.sub_temp.mean()),pred=float(z.W30G.mean()),bias=float(z.error.mean()),rmse=float(np.sqrt(np.mean(z.error**2)))))
  q['expert_min']=lo;q['expert_max']=hi;q['oracle_prediction']=oracle;hours.append(q)
  for (seed,context),z in t[(t.farm==f)&(t.day==day)&(t.member=='W30G')].groupby(['base_seed','context']):
   er=z.prediction.to_numpy()-z.sub_temp.to_numpy();robust.append(dict(farm=f,day=day,seed=int(seed),context=context,bias=float(er.mean()),rmse=float(np.sqrt(np.mean(er**2)))))
  ds=sorted(d.loc[f].index);fold=(ds.index(day)//5)%10
  z=np.load(R/f'집/코덱스/local/statistical_experiments_20261003_v1/T_DIAG10_{fold}_cpu.npz');keys=sorted({(s[:3],int(s[4:7])) for s in z['outer_train_id']});assert (f,day) not in keys
  tr=feats.loc[keys];med=tr.median();sd=tr.fillna(med).std(ddof=0).replace(0,1);v=feats.loc[(f,day)].fillna(med)
  candidates=[k for k in keys if k[0]==f and (k[1]>=179)==(day>=179)]
  dist=np.sqrt((((tr.loc[candidates].fillna(med)-v)/sd)**2).mean(axis=1))
  for rank,(nk,distance) in enumerate(dist.sort_values().head(5).items(),1):neighbors.append(dict(case_farm=f,case_day=day,fold=fold,rank=rank,neighbor_farm=nk[0],neighbor_day=nk[1],distance=float(distance),**d.loc[nk].to_dict()))
  origin=q[W].to_numpy()
  for nk in keys:
   nw=x[(x.farm==nk[0])&(x.day==nk[1])].sort_values('hour')[W].to_numpy()
   if np.array_equal(origin,nw):weather.append(dict(case_farm=f,case_day=day,fold=fold,train_farm=nk[0],train_day=nk[1],**d.loc[nk].to_dict()))
  idx=ds.index(day)
  for nd in ds[max(0,idx-4):idx+5]:timeline.append(dict(case_farm=f,case_day=day,farm=f,day=nd,offset_days=nd-day,in_outer_train=(f,nd) in keys,**d.loc[(f,nd)].to_dict()))
 for name,rows in [('stats',stats),('neighbors',neighbors),('same_weather_train',weather),('timeline',timeline),('robustness',robust)]:pd.DataFrame(rows).to_csv(H/(name+'.csv'),index=False,encoding='utf-8-sig')
 pd.concat(hours).to_csv(H/'hours.csv',index=False,encoding='utf-8-sig')
 d.reset_index().to_csv(H/'all_days.csv',index=False,encoding='utf-8-sig')
 (H/'result.json').write_text(json.dumps(dict(stats=stats,neighbors=neighbors,weather_train=weather,timeline=timeline,robustness=robust),ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(stats=stats,neighbors=neighbors,weather=weather,timeline=timeline,robustness=robust),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
