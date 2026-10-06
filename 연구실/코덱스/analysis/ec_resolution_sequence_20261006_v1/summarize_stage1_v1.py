from pathlib import Path
import json,sys,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
assert (H/'stage1_receipt_v2.json').exists(),'Do not score partial learning'
assert not (ROOT/'연구실/코덱스/local'/H.name/'stage1_v2/worker.lock').exists()
O=H/'stage1_summary_v1';O.mkdir(exist_ok=False)
a=pd.read_csv(H/'stage1_rows_v2.csv',float_precision='round_trip');assert len(a)==1152
av=a.groupby(['context','row_id','farm','day','hour'],as_index=False)[['sub_ec','prediction','raw_et','raw_lgb','raw_mlp','raw_pfn']].mean()
days=[]
for (ctx,f,day),q in av.groupby(['context','farm','day']):
    e=q.prediction-q.sub_ec
    days.append(dict(context=ctx,farm=f,day=int(day),ymean=float(q.sub_ec.mean()),pmean=float(q.prediction.mean()),bias=float(e.mean()),rmse=float(np.sqrt(np.dot(e,e)/len(e))),n=len(e)))
pd.DataFrame(days).to_csv(O/'day_scores.csv',index=False)
pairs=[]
for ctx,q in pd.DataFrame(days).groupby('context'):
    for f,good,bad in [('F47',160,161),('F13',98,112)]:
        x=q[(q.farm==f)&(q.day==good)].iloc[0];y=q[(q.farm==f)&(q.day==bad)].iloc[0]
        pairs.append(dict(context=ctx,farm=f,good=good,bad=bad,good_prediction=x.pmean,bad_prediction=y.pmean,prediction_difference=y.pmean-x.pmean,true_difference=y.ymean-x.ymean,bias_difference=y.bias-x.bias))
pd.DataFrame(pairs).to_csv(O/'same_model_pair_gaps.csv',index=False)
spread=[]
for (f,day),q in pd.DataFrame(days).groupby(['farm','day']):spread.append(dict(farm=f,day=int(day),minimum=float(q.pmean.min()),maximum=float(q.pmean.max()),training_range=float(q.pmean.max()-q.pmean.min()),prediction_std=float(q.pmean.std())))
pd.DataFrame(spread).to_csv(O/'training_variation.csv',index=False)
tree_rows=[]
L=ROOT/'연구실/코덱스/local'/H.name/'stage1_v2'
for p in L.glob('*_et_paths.json'):
    paths=json.loads(p.read_text(encoding='utf-8'));assert len(paths)==1200
    for f in ['F47','F13']:
        g=[x for x in paths if x['farm']==f];total=math.fsum(x['difference'] for x in g)/600
        features={}
        for x in g:
            sp=x['first_divergence'];name=sp['feature'] if sp else 'same_leaf'
            bucket=features.setdefault(name,[]);bucket.append(x['difference'])
        assert abs(sum(math.fsum(v)/600 for v in features.values())-total)<1e-12
        for name,v in features.items():tree_rows.append(dict(context=p.name.removesuffix('_et_paths.json'),farm=f,first_divergence=name,trees=len(v),signed_prediction_gap=math.fsum(v)/600,total_et_gap=total))
pd.DataFrame(tree_rows).to_csv(O/'first_tree_divergence.csv',index=False)
pd.DataFrame(av).to_csv(O/'hour_scores.csv',index=False)
(O/'completion.json').write_text(json.dumps(dict(status='COMPLETE_DIAGNOSTIC_SUMMARY',days=16,pairs=8,contexts=4,trees=4800,fit=0,adoption=False),indent=2),encoding='utf-8')
print(pd.DataFrame(days).to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))
