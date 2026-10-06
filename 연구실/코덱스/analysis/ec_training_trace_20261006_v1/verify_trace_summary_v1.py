from pathlib import Path
import csv,json,math,datetime
from collections import defaultdict
H=Path(__file__).resolve().parent;D=H/'results_v4';U=H/'summary_v1'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def same(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10),(a,b)
def avg(v):return math.fsum(v)/len(v)
raw=read(D/'support_profiles.csv');copy=read(U/'raw_support_profiles.csv');assert len(raw)==len(copy)
for a,b in zip(raw,copy):
    for k,v in a.items():
        if k in ['context','farm','level']:assert v==b[k]
        else:same(v,b[k])
days=read(D/'support_days.csv');tops=read(U/'top_support_days.csv');pool=defaultdict(list);selected=defaultdict(list)
for r in days:pool[r['context'],r['query_farm'],r['query_day'],r['level']].append(r)
for r in tops:selected[r['context'],r['query_farm'],r['query_day'],r['level']].append(r)
assert set(pool)==set(selected)
for key,q in pool.items():
    chosen=selected[key];assert len(chosen)==min(10,len(q));assert [int(r['rank']) for r in chosen]==list(range(1,len(chosen)+1));weights=[float(r['weight']) for r in chosen];assert all(a>=b-1e-12 for a,b in zip(weights,weights[1:]));assert min(weights)>=sorted([float(r['weight']) for r in q],reverse=True)[len(chosen)-1]-1e-12
    for r in chosen:
        ref=next(v for v in q if v['train_farm']==r['train_farm'] and v['train_day']==r['train_day'])
        for col in ['weight','contribution','weighted_ec']:same(r[col],ref[col])
def aggregate(source,cols,fields):
    g=defaultdict(lambda:defaultdict(list))
    for r in source:
        key=tuple(r[c] for c in cols)
        for field in fields:g[key][field].append(float(r[field]))
    return {key:{field:math.fsum(vals) for field,vals in values.items()} for key,values in g.items()}
for src,dest,cols,fields in [
 ('pair_weight_changes.csv','pair_day_changes.csv',['context','query_farm','level','farm','day'],['delta_weight','raw_contribution','centered_contribution']),
 ('weight_shapley_rows.csv','group_shapley_days.csv',['query_farm','group','farm','day'],['phi_weight','raw_contribution','centered_contribution'])]:
    expected=aggregate(read(D/src),cols,fields);output=read(U/dest);assert len(output)==len(expected)
    for r in output:
        values=expected[tuple(r[c] for c in cols)]
        for field,v in values.items():same(v,r[field])
split=read(D/'first_split_training.csv');groups=defaultdict(dict)
for r in split:groups[r['context'],r['farm'],r['feature']].setdefault(r['tree'],{})[r['side']]=r
associations=read(U/'first_branch_associations.csv');assert len(associations)==len(groups)
for r in associations:
    q=groups[r['context'],r['farm'],r['feature']];assert all(set(v)=={'good','bad'} for v in q.values());good=[float(v['good']['mean_ec']) for v in q.values()];bad=[float(v['bad']['mean_ec']) for v in q.values()];hi=[float(v['bad']['high_row_fraction'])-float(v['good']['high_row_fraction']) for v in q.values()]
    values=dict(trees=len(q),mean_good_branch_ec=avg(good),mean_bad_branch_ec=avg(bad),mean_branch_ec_difference=avg([b-a for a,b in zip(good,bad)]),mean_high_row_fraction_difference=avg(hi),positive_branch_ec_trees=sum(b>a for a,b in zip(good,bad)))
    for c,v in values.items():same(v,r[c])
out=dict(status='PASS_SUMMARY_TOP_SUPPORT_AND_AGGREGATION',new_fit=0,top_support_groups=len(pool),branch_association_groups=len(groups),scope='raw profile copy/top10 by mass/day delta and phi sums/paired firstbranch means; smoothed support independently verified by fulltree audit')
path=H/('verify_trace_summary_v1_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
