from models import *
from run_models import metrics
def main():
 world=joblib.load(O/'world.joblib');lab=world['lab'];fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==5);tm,vm=S.common.split_mask(lab,fd);tr,va=lab[tm],lab[vm];daily=tr.groupby(['farm','day']).agg(y=('sub_temp','mean'),air=('in_temp','mean'),n=('in_temp','count'));warm=daily[(daily.n==24)&(daily.y-daily.air>=2)].index;factor=np.array([20 if (f,int(d)) in warm else 1 for f,d in zip(tr.farm,tr.day)]);rows=[];fit=[]
 for seed in [7,101]:
  with S.threadpool_limits(limits=2):model=Temp().fit(tr,world['ct'],world['phc'],world['w'][tm]*factor,seed);joblib.dump(model,O/f'rare_weight20_{seed}.joblib');p=model.predict(va);q=tr[(tr.farm=='F47')&(tr.day==178)];pp=model.predict(q)
  for name in ['BASE','CODEX']:fit.append(dict(seed=seed,member=name,**metrics(pp[name],q.sub_temp.to_numpy())))
  ref=world['outer'];r=ref[(ref.validator=='DIAG10')&(ref.base_seed==seed)&(ref.context=='1-8')].pivot(index='row_id',columns='member',values='prediction').loc[va.row_id];g=S.gate(va);out=(.4+.1*g)*p['BASE']+(.6-.4*g)*p['CODEX']+.3*g*r.PFN.to_numpy()
  for f,day in [('F13',194),('F47',191),(None,None)]:
   m=np.ones(len(va),bool) if f is None else ((va.farm==f)&(va.day==day)).to_numpy();a=metrics(out[m],va.sub_temp.to_numpy()[m]);b=metrics(r.W30G.to_numpy()[m],va.sub_temp.to_numpy()[m]);rows.append(dict(seed=seed,farm=f,day=day,mean_shift=a['mean']-b['mean'],rmse_change=a['rmse']-b['rmse'],**a))
  print('RARE_WEIGHT',seed,rows[-2],flush=True)
 pd.DataFrame(rows).to_csv(H/'rare_weight_diagnostic.csv',index=False);pd.DataFrame(fit).to_csv(H/'rare_weight_training_fit.csv',index=False);savej(H/'rare_weight_days.json',dict(train_warm_days=[list(k) for k in warm],factor=20,query_label_in_fit=False))
if __name__=='__main__':main()
