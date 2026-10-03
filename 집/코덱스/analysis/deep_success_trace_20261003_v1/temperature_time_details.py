from models import *
from run_models import metrics
def main():
 world=joblib.load(O/'world.joblib');effects=pd.read_csv(H/'effects_v2.csv');groups=json.loads((H/'TEMP_groups.json').read_text(encoding='utf-8'));columns=groups['indoor_temperature'];sub={g:[] for g in ['current','midnight_anchor','within_day','continuous_history']}
 for c in columns:
  g='current' if c in ['in_temp','ph_in_temp_None'] else 'midnight_anchor' if 'h0' in c else 'within_day' if c.startswith('seg_') or '_reset' in c or c in ['in_temp_mean','in_temp_std','in_temp_diff','in_temp_tdmean','in_temp_tdsum'] else 'continuous_history';sub[g].append(c)
 rows=[]
 for case in [z for z in world['selected'] if z['target']=='TEMP' and (z['farm'],z['day']) in TEMP[:5]]:
  f,day,k=case['farm'],case['day'],case['fold'];fd=next(fd for n,fold,fd in world['folds'] if n=='DIAG10' and fold==k);tm,vm=S.common.split_mask(world['lab'],fd);tr,va=world['lab'][tm],world['lab'][vm];q=va[(va.farm==f)&(va.day==day)].sort_values('hour');y=q.sub_temp.to_numpy()
  for seed in [7,101]:
   model=joblib.load(O/f'TEMP_{k}_{seed}_model.joblib');r=world['outer'];r=r[(r.validator=='DIAG10')&(r.base_seed==seed)&(r.context=='1-8')&(r.farm==f)&(r.day==day)].pivot(index='row_id',columns='member',values='prediction').loc[q.row_id];g=S.gate(q);base=r.W30G.to_numpy()
   for e in effects[(effects.target=='TEMP')&(effects.farm==f)&(effects.day==day)&(effects.seed==seed)&(effects.group=='indoor_temperature')].itertuples():
    donor=tr[(tr.farm==f)&(tr.day==e.donor_day)].sort_values('hour')
    for name,cs in sub.items():
     ch=q.copy();ch[cs]=donor[cs].to_numpy()
     with S.threadpool_limits(limits=2):p=model.predict(ch)
     out=(.4+.1*g)*p['BASE']+(.6-.4*g)*p['CODEX']+.3*g*r.PFN.to_numpy();m=metrics(out,y);rows.append(dict(farm=f,day=day,seed=seed,group=name,donor_day=int(e.donor_day),mean_shift=float(np.mean(out-base)),rmse_change=m['rmse']-metrics(base,y)['rmse'],**m))
 pd.DataFrame(rows).to_csv(H/'temperature_time_details.csv',index=False);savej(H/'temperature_time_groups.json',sub);print(pd.DataFrame(rows).groupby(['farm','day','group'])[['rmse_change','mean_shift']].mean().to_string())
if __name__=='__main__':main()
