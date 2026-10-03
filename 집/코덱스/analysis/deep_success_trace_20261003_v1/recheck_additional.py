from models import *
from period_control import NoDayTemp
from recheck import mean,close
from run_models import metrics
def main():
 w=joblib.load(O/'world.joblib');lab=w['lab'];fd=next(fd for n,k,fd in w['folds'] if n=='DIAG10' and k==5);tm,vm=S.common.split_mask(lab,fd);tr,va=lab[tm],lab[vm];n=0
 for filename,pattern in [('failure_interventions.csv','intervention_{arm}_{seed}.joblib'),('period_control.csv','no_day_{seed}.joblib'),('rare_weight_diagnostic.csv','rare_weight20_{seed}.joblib')]:
  df=pd.read_csv(H/filename,float_precision='round_trip')
  for row in df.itertuples():
   model=joblib.load(O/pattern.format(arm=getattr(row,'arm',''),seed=row.seed));q=va if pd.isna(row.day) else va[(va.farm==row.farm)&(va.day==int(row.day))]
   with S.threadpool_limits(limits=2):p=model.predict(q)
   ref=w['outer'];r=ref[(ref.validator=='DIAG10')&(ref.base_seed==row.seed)&(ref.context=='1-8')].pivot(index='row_id',columns='member',values='prediction').loc[q.row_id];g=S.gate(q);pred=(.4+.1*g)*p['BASE']+(.6-.4*g)*p['CODEX']+.3*g*r.PFN.to_numpy();e=pred-q.sub_temp.to_numpy();close(row.mean,mean(pred));close(row.bias,mean(e));close(row.rmse,np.sqrt(mean(e*e)));close(row.mean_shift,mean(pred)-mean(r.W30G.to_numpy()));n+=4
 phases=pd.read_csv(H/'event_temperature_phases.csv',float_precision='round_trip');raw=pd.read_csv(O/'temperature_raw_public.csv',float_precision='round_trip')
 for row in phases.itertuples():
  q=raw[(raw.farm==row.farm)&(raw.day==row.day)];mask=q.hour<=6 if row.segment=='night0_6' else q.hour.between(7,16) if row.segment=='day7_16' else q.hour>=17;q=q[mask];close(row.gap,mean(q.sub_temp-q.in_temp));close(row.bias,mean(q.prediction-q.sub_temp));n+=2
 detail=pd.read_csv(H/'temperature_time_details.csv',float_precision='round_trip');gg=json.loads((H/'temperature_time_groups.json').read_text(encoding='utf-8'));allg=json.loads((H/'TEMP_groups.json').read_text(encoding='utf-8'))['indoor_temperature'];assert sorted(c for v in gg.values() for c in v)==sorted(allg)
 for key,q in detail.groupby(['farm','day','group']):assert len(q)==10;n+=1
 for row in pd.read_csv(H/'warm_178_training_fit.csv',float_precision='round_trip').itertuples():
  if row.part not in ['BASE','CODEX']:continue
  q=tr[(tr.farm=='F47')&(tr.day==178)];model=joblib.load(O/f'TEMP_5_{row.seed}_model.joblib')
  with S.threadpool_limits(limits=2):p=model.predict(q)[row.part]
  e=p-q.sub_temp.to_numpy();close(row.mean,mean(p));close(row.rmse,np.sqrt(mean(e*e)));n+=2
 savej(H/'additional_verification.json',dict(status='PASS',checks=n,training_178_in_actual_fit=bool(((tr.farm=='F47')&(tr.day==178)).sum()==24),notes='Recomputed interventions/phase arithmetic and rare training fit from persisted models; statistics are diagnostic, not new validation.'))
 print('ADDITIONAL_PASS',n,flush=True)
if __name__=='__main__':main()
