from models import *
from run_models import metrics
class NoDayTemp(Temp):
 def fit(self,tr,ct,phc,w,seed):
  old=S.FEATURE_COLUMNS;self.reduced=[c for c in old if c not in ['day','second']];S.FEATURE_COLUMNS=self.reduced
  try:super().fit(tr,[c for c in ct if c not in ['day','second']],phc,w,seed)
  finally:S.FEATURE_COLUMNS=old
  return self
 def predict(self,q):
  old=S.FEATURE_COLUMNS;S.FEATURE_COLUMNS=self.reduced
  try:return super().predict(q)
  finally:S.FEATURE_COLUMNS=old
def main():
 world=joblib.load(O/'world.joblib');lab=world['lab'];fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==5);tm,vm=S.common.split_mask(lab,fd);tr,va=lab[tm],lab[vm];rows=[];trainrows=[]
 for seed in [7,101]:
  with S.threadpool_limits(limits=2):
   original=joblib.load(O/f'TEMP_5_{seed}_model.joblib');q=tr[(tr.farm=='F47')&(tr.day==178)];p=original.predict(q)
   for name,values in p.items():trainrows.append(dict(seed=seed,farm='F47',day=178,part=name,n=len(q),**metrics(values,q.sub_temp.to_numpy())))
   model=NoDayTemp().fit(tr,world['ct'],world['phc'],world['w'][tm],seed);joblib.dump(model,O/f'no_day_{seed}.joblib');p=model.predict(va)
  ref=world['outer'];r=ref[(ref.validator=='DIAG10')&(ref.base_seed==seed)&(ref.context=='1-8')].pivot(index='row_id',columns='member',values='prediction').loc[va.row_id];g=S.gate(va);out=(.4+.1*g)*p['BASE']+(.6-.4*g)*p['CODEX']+.3*g*r.PFN.to_numpy()
  for f,day in [('F13',194),('F47',191),(None,None)]:
   m=np.ones(len(va),bool) if f is None else ((va.farm==f)&(va.day==day)).to_numpy();a=metrics(out[m],va.sub_temp.to_numpy()[m]);b=metrics(r.W30G.to_numpy()[m],va.sub_temp.to_numpy()[m]);rows.append(dict(seed=seed,farm=f,day=day,mean_shift=a['mean']-b['mean'],rmse_change=a['rmse']-b['rmse'],**a))
  print('NO_DAY',seed,rows[-2],flush=True)
 pd.DataFrame(rows).to_csv(H/'period_control.csv',index=False);pd.DataFrame(trainrows).to_csv(H/'warm_178_training_fit.csv',index=False)
if __name__=='__main__':main()
