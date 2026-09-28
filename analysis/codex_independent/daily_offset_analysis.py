# -*- coding: utf-8 -*-
import sys, warnings, json
sys.dont_write_bytecode=True
from independent_analysis import *
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
tx,ty,sx=common.load_raw(); z=features(pd.concat([tx,sx],ignore_index=True))
saved=np.load(ROOT/'research/local/eval_v6_oof.npz',allow_pickle=True)
d=z.merge(ty[['row_id','sub_temp']],on='row_id').merge(pd.DataFrame({'row_id':saved['row_id'],'baseline':saved['F60ND__DIAG10']}),on='row_id')
d=d[d.sub_temp.notna()&np.isfinite(d.baseline)].reset_index(drop=True)
y=d.sub_temp.to_numpy(); b=d.baseline.to_numpy()
cols=[c for c in z if c not in ['row_id','farm','t'] and '_hist' not in c and '_seam' not in c]
pc=['farm_id','sin','cos','in_temp','out_temp','out_rad','act_heating']+[c for c in z if '_reset' in c]
pred={n:np.full(len(d),np.nan) for n in ['daily_reset','daily_h0']}
for k,fd in enumerate(folds(d)):
    tr,va=common.split_mask(d,fd)
    lin=ridge().fit(d.loc[tr,pc],y[tr]); base_tr=lin.predict(d.loc[tr,pc]); base_va=lin.predict(d.loc[va,pc])
    train=d.loc[tr].copy(); train['res']=y[tr]-base_tr
    daily=train.groupby(['farm','day']).res.transform('mean').to_numpy()
    for name in pred:
        cc=cols if name=='daily_reset' else ['farm_id','day','second']+[c for c in cols if c.endswith('_h0')]
        model=lgb().fit(train[cc],daily)
        pred[name][va]=base_va+model.predict(d.loc[va,cc])
    print('daily fold',k+1,flush=True)
res={}
for name,p in pred.items():
    res[name]=compare(d,y,b,p)
    res[name+'_blend20']=compare(d,y,b,.8*b+.2*p)
np.savez_compressed(HERE/'daily_offset_oof.npz',row_id=d.row_id.values,**pred)
with open(HERE/'daily_results.json','w',encoding='utf-8') as f: json.dump(res,f,indent=2,ensure_ascii=False)
print(json.dumps(res),flush=True)

