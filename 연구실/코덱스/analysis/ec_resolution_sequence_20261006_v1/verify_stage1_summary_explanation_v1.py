"""Independent summary and stored exact-coalition arithmetic audit; no fitting."""
from pathlib import Path
import csv,json,math,ast,datetime
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3];R=H/'stage1_summary_v1';E=H/'stage1_explanation_v1';L=ROOT/'연구실/코덱스/local'/H.name/'stage1_v2'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def avg(v):return math.fsum(map(float,v))/len(v)
def same(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10),(a,b)
source=read(H/'stage1_rows_v2.csv');byrow=defaultdict(list)
for r in source:byrow[r['context'],r['row_id']].append(r)
days=defaultdict(list)
for (context,rid),q in byrow.items():assert len(q)==3;days[context,q[0]['farm'],int(q[0]['day'])].append(dict(y=float(q[0]['sub_ec']),p=avg([r['prediction'] for r in q])))
scores={}
for key,q in days.items():
    e=[r['p']-r['y'] for r in q];assert len(e)==24;scores[key]=dict(ymean=avg([r['y'] for r in q]),pmean=avg([r['p'] for r in q]),bias=avg(e),rmse=math.sqrt(avg([x*x for x in e])),n=24)
assert len(scores)==16
for r in read(R/'day_scores.csv'):
    q=scores[r['context'],r['farm'],int(r['day'])]
    for col,value in q.items():same(value,r[col])
for r in read(R/'same_model_pair_gaps.csv'):
    a=scores[r['context'],r['farm'],int(r['good'])];b=scores[r['context'],r['farm'],int(r['bad'])]
    for col,value in dict(good_prediction=a['pmean'],bad_prediction=b['pmean'],prediction_difference=b['pmean']-a['pmean'],true_difference=b['ymean']-a['ymean'],bias_difference=b['bias']-a['bias']).items():same(value,r[col])
for r in read(R/'training_variation.csv'):
    vals=[q['pmean'] for (context,farm,day),q in scores.items() if farm==r['farm'] and day==int(r['day'])];mean=avg(vals)
    for col,value in dict(minimum=min(vals),maximum=max(vals),training_range=max(vals)-min(vals),prediction_std=math.sqrt(math.fsum((v-mean)**2 for v in vals)/(len(vals)-1))).items():same(value,r[col])
treebuckets={}
for p in L.glob('*_et_paths.json'):
    paths=json.loads(p.read_text(encoding='utf-8'));context=p.name.removesuffix('_et_paths.json')
    for farm in ['F13','F47']:
        q=[r for r in paths if r['farm']==farm];assert len(q)==600;total=avg([r['difference'] for r in q]);buckets=defaultdict(list)
        for r in q:buckets[r['first_divergence']['feature'] if r['first_divergence'] else 'same_leaf'].append(r['difference'])
        for feature,vals in buckets.items():treebuckets[context,farm,feature]=dict(trees=len(vals),signed_prediction_gap=math.fsum(vals)/600,total_et_gap=total)
for r in read(R/'first_tree_divergence.csv'):
    bucket=treebuckets.pop((r['context'],r['farm'],r['first_divergence']))
    for col,value in bucket.items():same(value,r[col])
assert not treebuckets
contributions=read(E/'grouped_contributions.csv');models=json.loads((E/'verification.json').read_text(encoding='utf-8'))['models'];summary=[]
for model in models:
    kind,farm=model['model'],model['farm'];rows=[r for r in contributions if r['model']==kind and r['farm']==farm];n=len(rows);assert n==model['n_groups']==5
    vals={int(r['mask']):float(r['prediction']) for r in read(E/f'{kind}_{farm}_subsets.csv')};assert set(vals)==set(range(1<<n));calculated=[]
    for i,row in enumerate(rows):
        terms=[]
        for mask in range(1<<n):
            if not mask&(1<<i):k=mask.bit_count();w=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n);terms.append(w*(vals[mask|(1<<i)]-vals[mask]))
        value=math.fsum(terms);same(value,row['prediction_change']);same(value*float(row['mix_weight']),row['mix_change']);assert len(ast.literal_eval(row['columns']))>=1;calculated.append(value)
    rawname='raw_et' if kind=='ET' else 'raw_lgb';good=rows[0]['good'];bad=rows[0]['bad']
    for day,mask in [(good,0),(bad,(1<<n)-1)]:
        endpoint=next(r for r in source if r['context']=='common_intersection' and r['seed']=='7' and r['farm']==farm and r['day']==day and r['hour']=='0');same(vals[mask],endpoint[rawname])
    same(vals[0],model['good_prediction']);same(vals[(1<<n)-1],model['bad_prediction']);same(math.fsum(calculated),vals[(1<<n)-1]-vals[0]);same(math.fsum(calculated),model['sum_contributions']);summary.append(dict(model=kind,farm=farm,coalitions=len(vals),delta=vals[(1<<n)-1]-vals[0]))
out=dict(status='PASS_SUMMARY_AND_STORED_COALITION_ARITHMETIC',daily_scores=16,same_model_pairs=8,training_variation_targets=4,et_trace_pairs=8,shapley_models=summary,fit=0,limitations=['Shapley recomputed from stored32 coalition predictions; model hybrid predictions not independently re-fit','first split buckets are tree-gap accounting, not causal feature attribution','seed7 common training ET/LGB at0h only; PFN/MLP and daily ensemble not explained'],severe_counts_by_target=[dict(farm=farm,day=day,non_success=sum(q['rmse']>.1 for (c,f,d),q in scores.items() if f==farm and d==day),severe=sum(abs(q['bias'])>=.2 for (c,f,d),q in scores.items() if f==farm and d==day)) for farm,day in [('F13',98),('F13',112),('F47',160),('F47',161)]])
path=H/('verify_stage1_summary_explanation_v1_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
