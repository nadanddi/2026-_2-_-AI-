from pathlib import Path
import sys,json,math,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));import stage1_v2 as S
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
assert (H/'stage1_receipt_v2.json').exists(),'Finish original four conditions before explanation'
assert not (S.O/'worker.lock').exists()
R=H/'stage1_explanation_v1';R.mkdir(exist_ok=False)
_,jobs,p=S.prepare();t,q=jobs['common_intersection'];names=['season']+S.M.RAW
out=[];tables=[]
for kind,model,cols,weight in [('ET',S.M.et(7),S.M.FULL,.48),('LGB',S.M.lg(7),S.M.BASE,.24)]:
    with threadpool_limits(limits=2):
        model.fit(t[cols],t.sub_ec.to_numpy(float));actual=np.asarray(model.predict(q[cols]),float)
        with np.load(S.O/'common_intersection_r3_7.npz',allow_pickle=False) as z:expected=z['raw_et' if kind=='ET' else 'raw_lgb']
        assert np.max(abs(actual-expected))<1e-8
        for farm,good,bad in [('F47',160,161),('F13',98,112)]:
            a=q[(q.farm==farm)&(q.day==good)&(q.hour==0)].iloc[0];b=q[(q.farm==farm)&(q.day==bad)&(q.hour==0)].iloc[0]
            groups={n:[c for c in cols if c==n or c.startswith(n+'_')] for n in names}
            changing=[n for n in names if groups[n] and not np.array_equal(a[groups[n]].to_numpy(float),b[groups[n]].to_numpy(float))]
            untouched=[c for c in cols if not any(c in groups[n] for n in changing)]
            assert np.array_equal(a[untouched].to_numpy(float),b[untouched].to_numpy(float))
            n=len(changing);frames=[]
            for mask in range(1<<n):
                row=a[cols].copy()
                for i,g in enumerate(changing):
                    if mask&(1<<i):row[groups[g]]=b[groups[g]]
                frames.append(row)
            f=pd.DataFrame(frames,columns=cols).astype(float);values=np.asarray(model.predict(f),float)
            contrib=[]
            for i,g in enumerate(changing):
                terms=[]
                for mask in range(1<<n):
                    if mask&(1<<i):continue
                    k=mask.bit_count();w=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
                    terms.append(w*(values[mask|(1<<i)]-values[mask]))
                val=math.fsum(terms);contrib.append(val)
                out.append(dict(model=kind,farm=farm,good=good,bad=bad,hour=0,group=g,columns=groups[g],prediction_change=val,mix_weight=weight,mix_change=weight*val,good_value=float(a[g]),bad_value=float(b[g]),definition='exact interventional grouped Shapley along two model inputs; model arithmetic not physical causal effect'))
            assert abs(sum(contrib)-(values[-1]-values[0]))<1e-10
            tables.append(dict(model=kind,farm=farm,n_groups=n,subsets=len(values),good_prediction=float(values[0]),bad_prediction=float(values[-1]),total_difference=float(values[-1]-values[0]),sum_contributions=sum(contrib),max_cache_diff=float(np.max(abs(actual-expected)))))
            pd.DataFrame(dict(mask=range(len(values)),prediction=values)).to_csv(R/f'{kind}_{farm}_subsets.csv',index=False)
    del model;gc.collect()
pd.DataFrame(out).to_csv(R/'grouped_contributions.csv',index=False)
(R/'verification.json').write_text(json.dumps(dict(status='PASS_FUNCTION_DECOMPOSITION',models=tables,fit=2,season_transformation='common training recipe unchanged',posthoc=True,physical_causality=False,PFN_explained=False),indent=2),encoding='utf-8')
print(pd.DataFrame(out)[['model','farm','group','prediction_change','mix_change']].to_string(index=False))
