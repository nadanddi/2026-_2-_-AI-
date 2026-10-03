from models import *
from run_models import metrics
def main():
 world=joblib.load(O/'world.joblib');lab=world['lab'];fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==5);tm,vm=S.common.split_mask(lab,fd);tr=lab[tm].copy();va=lab[vm].copy();rows=[];parts=[]
 for seed in [7,101]:
  with S.threadpool_limits(limits=2):
   for arm in ['drop_F47_178','uniform_weights']:
    keep=~((tr.farm=='F47')&(tr.day==178)) if arm=='drop_F47_178' else np.ones(len(tr),bool);weight=world['w'][tm][keep] if arm=='drop_F47_178' else np.ones(keep.sum())
    model=Temp().fit(tr[keep],world['ct'],world['phc'],weight,seed);joblib.dump(model,O/f'intervention_{arm}_{seed}.joblib');p=model.predict(va)
    ref=world['outer'];r=ref[(ref.validator=='DIAG10')&(ref.base_seed==seed)&(ref.context=='1-8')&(ref.member.isin(['W30G','PFN']))].pivot(index='row_id',columns='member',values='prediction').loc[va.row_id]
    g=S.gate(va);out=(.4+.1*g)*p['BASE']+(.6-.4*g)*p['CODEX']+.3*g*r.PFN.to_numpy();y=va.sub_temp.to_numpy()
    for f,day in [('F13',194),('F47',191),(None,None)]:
     m=np.ones(len(va),bool) if f is None else ((va.farm==f)&(va.day==day)).to_numpy();stats=metrics(out[m],y[m]);base=metrics(r.W30G.to_numpy()[m],y[m]);rows.append(dict(arm=arm,seed=seed,farm=f,day=day,n=int(m.sum()),rmse_change=stats['rmse']-base['rmse'],mean_shift=stats['mean']-base['mean'],**stats))
    print('INTERVENTION',seed,arm,rows[-2],flush=True)
 pd.DataFrame(rows).to_csv(H/'failure_interventions.csv',index=False)
if __name__=='__main__':main()
