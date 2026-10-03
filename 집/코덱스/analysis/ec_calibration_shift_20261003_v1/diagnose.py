from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
lab,core,wv,folds,outer=S.loadec();records=[];checks=0
with threadpool_limits(limits=2):
 for v,k,tm,vm in folds:
  tr=lab[tm];va=lab[vm].reset_index(drop=True)
  z=dict(np.load(S.OUT/f'E_{v}_{k}_cpu.npz'));b=lab.set_index('row_id').reindex(z['row_id']).reset_index()
  assert set(z['row_id']).isdisjoint(z['inner_train_id']) and set(z['row_id'])<=set(tr.row_id);checks+=1
  bag=np.mean([np.load(S.OUT/f'E_{v}_{k}_pfn_{i}.npz')['prediction'] for i in [1,2,3,4]],axis=0)
  for seed in [7,101,2024]:
   ib=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi'])
   ob=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
   for tag,df,pred in [('inner',b,ib),('outer',va,ob)]:
    dayy=df.groupby(['farm','day']).sub_ec.mean()
    for hour in [0,6,23]:
     x=df.loc[df.hour<=hour,['farm','day']].assign(pred=pred[df.hour<=hour]).groupby(['farm','day']).pred.mean();yy=dayy.reindex(x.index)
     for group,m in [('all',np.ones(len(x),dtype=bool)),('true_high',(yy>=1).to_numpy()),('pred_high',(x>=1).to_numpy())]:
      a=x.to_numpy()[m];y=yy.to_numpy()[m];n=len(a)
      if n:
       bias=math.fsum(float(p)-float(t) for p,t in zip(a,y))/n;assert abs(bias-np.mean(a-y))<1e-12;checks+=1
       rmse=math.sqrt(math.fsum((float(p)-float(t))**2 for p,t in zip(a,y))/n);assert abs(rmse-np.sqrt(np.mean((a-y)**2)))<1e-12;checks+=1
      else:bias=rmse=np.nan
      records.append(dict(validator=v,fold=k,seed=seed,sample=tag,hour=hour,group=group,n=n,bias=bias,rmse=rmse,train_days=len(z['inner_train_id'])//24 if tag=='inner' else len(tr)//24))
  print(v,k,'done',flush=True)
o=pd.DataFrame(records);o.to_csv(H/'distribution_v1.csv',index=False)
# Group pooled bias, not average fold bias. Days recur across nested folds; not independent sample sizes.
summary=o.groupby(['validator','sample','hour','group']).apply(lambda d:pd.Series({'observations':int(d.n.sum()),'weighted_bias':float(np.nansum(d.bias*d.n)/d.n.sum()) if d.n.sum() else np.nan,'pooled_rmse':float(np.sqrt(np.nansum(d.rmse**2*d.n)/d.n.sum())) if d.n.sum() else np.nan}),include_groups=False).reset_index()
summary.to_csv(H/'summary_v1.csv',index=False)
paired=o[o.group=='pred_high'].pivot(index=['validator','fold','seed','hour'],columns='sample',values=['bias','n']).dropna()
paired.to_csv(H/'paired_pred_high_v1.csv')
(H/'verification_v1.json').write_text(json.dumps({'status':'PASS','checks':checks,'rows':len(o),'warning':'diagnostic only; nested day records repeat; no correction threshold tuned; inner training less than outer; level bias uses actual whole-day mean and causal predicted prefix; descriptive comparison cannot isolate cause'},ensure_ascii=False,indent=2),encoding='utf-8')
print(summary[summary.hour==0].to_string(index=False))
