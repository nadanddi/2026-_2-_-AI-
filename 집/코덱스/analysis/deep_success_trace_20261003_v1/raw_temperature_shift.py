from models import *
from run_models import metrics
def main():
 world=joblib.load(O/'world.joblib');original_loader=S.safeload;rows=[]
 for delta in [-3.,3.]:
  def modified():
   tx,ty,sx=original_loader();m=((tx.farm=='F47')&(tx.day==191))|((tx.farm=='F13')&(tx.day==194));tx.loc[m,'in_temp']=tx.loc[m,'in_temp']+delta;return tx,ty,sx
  S.safeload=modified
  try:lab,_,_,_,_,_=S.loadtemp()
  finally:S.safeload=original_loader;S.common.load_raw=original_loader
  for seed in [7,101]:
   model=joblib.load(O/f'TEMP_5_{seed}_model.joblib')
   for f,day in [('F13',194),('F47',191)]:
    q=lab[(lab.farm==f)&(lab.day==day)].sort_values('hour');old=world['lab'][(world['lab'].farm==f)&(world['lab'].day==day)].sort_values('hour');assert np.array_equal(q.row_id,old.row_id);assert np.array_equal(q.sub_temp,old.sub_temp)
    r=world['outer'];r=r[(r.validator=='DIAG10')&(r.base_seed==seed)&(r.context=='1-8')&(r.farm==f)&(r.day==day)].pivot(index='row_id',columns='member',values='prediction').loc[q.row_id]
    with S.threadpool_limits(limits=2):p=model.predict(q)
    g=S.gate(q);out=(.4+.1*g)*p['BASE']+(.6-.4*g)*p['CODEX']+.3*g*r.PFN.to_numpy();stats=metrics(out,q.sub_temp.to_numpy());rows.append(dict(seed=seed,farm=f,day=day,raw_shift=delta,mean_shift=stats['mean']-float(r.W30G.mean()),**stats))
  print('RAW_SHIFT_DONE',delta,flush=True)
 pd.DataFrame(rows).to_csv(H/'raw_temperature_shift.csv',index=False)
if __name__=='__main__':main()
