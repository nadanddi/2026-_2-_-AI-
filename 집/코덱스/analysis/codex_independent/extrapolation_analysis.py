# -*- coding: utf-8 -*-
import sys,json
sys.dont_write_bytecode=True
from independent_analysis import *
import features_v4 as F4
tx,ty,sx=common.load_raw(); z=features(pd.concat([tx,sx],ignore_index=True))
saved=np.load(ROOT/'research/local/eval_v6_oof.npz',allow_pickle=True)
d=z.merge(ty[['row_id','sub_temp']],on='row_id').merge(pd.DataFrame({'row_id':saved['row_id']}),on='row_id')
d=d[d.sub_temp.notna()].reset_index(drop=True)
ph=F4.phys_features().set_index('row_id').loc[d.row_id,'ph_in_temp_3'].to_numpy()
cols=[c for c in z if c not in ['row_id','farm','t'] and '_hist' not in c and '_seam' not in c]
pc=['farm_id','sin','cos','in_temp','out_temp','out_rad','act_heating']+[c for c in z if '_reset' in c]
y=d.sub_temp.to_numpy(); out={}
for th in [8,10,12]:
    days=d.loc[ph<th,['farm','day']].drop_duplicates()
    fd={f:set(days.loc[days.farm==f,'day']) for f in ['F13','F47']}
    tr,va=common.split_mask(d,fd)
    base=pd.Series(saved['F60ND__EXT'+str(th)],index=saved['row_id']).loc[d.row_id].to_numpy()
    assert np.array_equal(np.isfinite(base),va)
    lin=ridge().fit(d.loc[tr,pc],y[tr])
    model=lgb().fit(d.loc[tr,cols],y[tr]-lin.predict(d.loc[tr,pc]))
    p=lin.predict(d.loc[va,pc])+model.predict(d.loc[va,cols])
    out[str(th)]=compare(d.loc[va].reset_index(drop=True),y[va],base[va],.8*base[va]+.2*p)
    out[str(th)]['train_rows']=int(tr.sum()); out[str(th)]['val_days']=len(days)
    print(th,json.dumps(out[str(th)]),flush=True)
with open(HERE/'extrapolation_results.json','w',encoding='utf-8') as f: json.dump(out,f,indent=2,ensure_ascii=False)
