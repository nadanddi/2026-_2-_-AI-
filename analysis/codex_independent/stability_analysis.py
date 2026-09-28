# -*- coding: utf-8 -*-
import sys,json
sys.dont_write_bytecode=True
from independent_analysis import *
d=pd.read_json(HERE/'sub_temp_diagnostic.json'); pack=np.load(HERE/'sub_temp_independent_oof.npz',allow_pickle=True)
assert np.array_equal(d.row_id,pack['row_id'])
y=d.sub_temp.to_numpy(); b=d.baseline.to_numpy(); p=.8*b+.2*pack['resid_reset']; d['in_temp']=99.
out={}
for name,m in [('F13',d.farm.to_numpy()=='F13'),('F47',d.farm.to_numpy()=='F47'),('second',d.day.to_numpy()>=179)]:
    out[name]=compare(d.loc[m].reset_index(drop=True),y[m],b[m],p[m])
out['folds']=[]
for fd in folds(d):
    _,m=common.split_mask(d,fd)
    out['folds'].append({'rows':int(m.sum()),'base':rmse(y[m],b[m]),'blend':rmse(y[m],p[m]),'delta':rmse(y[m],p[m])-rmse(y[m],b[m])})
with open(HERE/'stability_results.json','w',encoding='utf-8') as f: json.dump(out,f,indent=2,ensure_ascii=False)
print(json.dumps(out))
