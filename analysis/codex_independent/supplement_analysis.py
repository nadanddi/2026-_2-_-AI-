# -*- coding: utf-8 -*-
import sys, json
sys.dont_write_bytecode=True
from independent_analysis import *
from sklearn.metrics import average_precision_score, roc_auc_score
out={}
for target in ['sub_temp','sub_ec']:
    d=pd.read_json(HERE/(target+'_diagnostic.json'))
    pack=np.load(HERE/(target+'_independent_oof.npz'),allow_pickle=True)
    assert np.array_equal(pack['row_id'],d.row_id)
    pp={k:pack[k] for k in pack.files if k!='row_id'}
    if target=='sub_temp':
        extra=np.load(HERE/'daily_offset_oof.npz',allow_pickle=True)
        assert np.array_equal(extra['row_id'],d.row_id)
        pp.update({k:extra[k] for k in extra.files if k!='row_id'})
    y=d[target].to_numpy(); b=d.baseline.to_numpy(); res={}
    d['in_temp']=99. # compare의 저온 분할은 이 보조 분석에서 사용하지 않음.
    for name,p in pp.items():
        if target=='sub_temp':
            gate=np.where(d.hour.to_numpy()<4,.8*b+.2*p,b)
            full=.8*b+.2*p
            r={'early_only20':compare(d,y,b,gate),'day_error':{}}
            for nm,values in [('baseline',b),('blend20',full),('early_only20',gate)]:
                e=values-y
                dm=pd.Series(e).groupby([d.farm,d.day]).transform('mean').to_numpy()
                r['day_error'][nm]={'level_rmse':float(np.sqrt(np.mean(dm**2))),'within_rmse':rmse(e,dm)}
            for nm,m in [('early',d.hour.to_numpy()<4),('second',d.day.to_numpy()>=179)]:
                r[nm+'_gated_ci']=compare(d.loc[m].reset_index(drop=True),y[m],b[m],gate[m])
            res[name]=r
        else:
            res[name]={}
            dayy=d.groupby(['farm','day'])[target].transform('mean').to_numpy()>1.5
            for h in [0,3,12,23]:
                m=d.hour.to_numpy()==h
                res[name][str(h)]={'n_days':int(m.sum()),'high_days':int(dayy[m].sum()),'ap':float(average_precision_score(dayy[m],p[m])),'auc':float(roc_auc_score(dayy[m],p[m])),'baseline_ap':float(average_precision_score(dayy[m],b[m]))}
    out[target]=res
_,_,test=common.load_raw()
out['test_counts']=test.groupby('farm').agg(rows=('row_id','size'),days=('day','nunique')).to_dict('index')
with open(HERE/'supplement_results.json','w',encoding='utf-8') as f: json.dump(out,f,indent=2,ensure_ascii=False)
print(json.dumps(out),flush=True)
