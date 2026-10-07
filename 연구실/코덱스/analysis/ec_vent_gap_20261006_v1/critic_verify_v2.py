from pathlib import Path
import csv,json,math,sys
from collections import defaultdict
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def avg(a):return math.fsum(a)/len(a)
def same(a,b):assert math.isclose(a,b,abs_tol=1e-11,rel_tol=1e-11),(a,b)
def flag(r,c):return r[c]=='True'
d=read(H/'results_v2/days.csv')
for r in d:
    for c in ['day','fold','n','sse','bias','sse_day','sse_shape','ymean','pmean','act_circfan_mean','in_temp_mean']:r[c]=float(r[c])
    r['fanlow']=r['act_circfan_mean']<10
groups=defaultdict(list)
for r in d:
    groups[(flag(r,'closed'),r['fanlow'])].append(r)
states=read(H/'results_v3/state_interaction.csv')
for r in states:
    q=groups[(flag(r,'closed'),flag(r,'fanlow'))]
    if r['ordinary']=='True':q=[t for t in q if flag(t,'ordinary')]
    same(len(q),float(r['n']));same(sum(t['ymean']>=1 for t in q),float(r['high']))
    same(math.sqrt(math.fsum(t['sse'] for t in q)/(24*len(q))),float(r['rmse']))
    same(sum(t['ymean']<1 and t['pmean']>=1 for t in q),float(r['true_falsehigh']))
budget=json.loads((H/'results_v3/gap_budget.json').read_text(encoding='utf-8'))
c=[r for r in d if flag(r,'ordinary') and flag(r,'closed')];r=[t for t in d if flag(t,'ordinary') and not flag(t,'closed')]
parts={
'constant_bias':avg([t['bias'] for t in c])**2-avg([t['bias'] for t in r])**2,
'daily_bias_variance':avg([(t['bias']-avg([u['bias'] for u in c]))**2 for t in c])-avg([(t['bias']-avg([u['bias'] for u in r]))**2 for t in r]),
'within_day_shape':avg([t['sse_shape']/24 for t in c])-avg([t['sse_shape']/24 for t in r])}
gap=avg([t['sse']/24 for t in c])-avg([t['sse']/24 for t in r])
for k,v in parts.items():same(v,budget['parts'][k]);same(v/gap,budget['fractions'][k])
trim=read(H/'results_v3/equal_fraction_trim.csv')
for t in trim:
    q=sorted(c if t['group']=='closed' else r,key=lambda z:(-z['sse'],z['farm'],z['day']))
    k=math.floor(len(q)*float(t['fraction']));same(k,float(t['removed']))
    same(math.sqrt(avg([u['sse']/24 for u in q[k:]])),float(t['rmse']))
matches=read(H/'results_v2/matched_pairs.csv');uniq=json.loads((H/'results_v4/unique_match_sensitivity.json').read_text(encoding='utf-8'))
for t in uniq:
    q=[u for u in matches if u['kind']==t['kind']];chosen={}
    for u in sorted(q,key=lambda v:float(v['distance'])):chosen.setdefault((u['farm'],u['rest_day']),u)
    same(len(chosen),t['pairs']);same(math.sqrt(avg([float(u['closed_mse']) for u in chosen.values()])),t['closed_rmse']);same(math.sqrt(avg([float(u['rest_mse']) for u in chosen.values()])),t['rest_rmse'])
context=read(H/'results_v4/state_target_context.csv')
for t in context:
    q=[u for u in d if flag(u,'closed')==flag(t,'closed') and flag(u,'ordinary')==flag(t,'ordinary')]
    same(len(q),float(t['days']));same(avg([u['bias'] for u in q]),float(t['bias']));same(math.sqrt(avg([u['sse']/24 for u in q])),float(t['rmse']))
prefix=read(H/'results_v3/prefix_cases.csv');onset=read(H/'results_v4/falsehigh_onset.csv')
for t in onset:
    q=[u for u in prefix if u['group']==t['group'] and u['hour']==t['hour']]
    same(len(q),float(t['days']));same(sum(flag(u,'selected1') for u in q),float(t['selected1']));same(sum(flag(u,'selected09') for u in q),float(t['selected09']))
# Independently verify complete training-day coverage directly from all 30 caches.
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
publickeys={(u['farm'],int(u['day'])) for u in d};coverage=[]
oof=read(ROOT/'연구실/코덱스/local'/H.name/'inputs/oof.csv')
oof={(u['seed'],u['fold'],u['row_id']):u for u in oof if u['validator']=='DIAG10'}
for fold in range(10):
    sets=[]
    for seed in [7,101,2024]:
        with np.load(ROOT/'연구실/코덱스/local'/H.name/'inputs/components'/f'DIAG10_{fold}_r3_{seed}.npz',allow_pickle=False) as z:
            ids=set(map(str,z['train_row_id']));keys={(i[:3],int(i[4:7])) for i in ids};assert keys<=publickeys;assert len(ids)==24*len(keys)
            assert keys=={k for k in publickeys if k in keys};sets.append(ids)
            for j,rid in enumerate(z['row_id'].astype(str)):
                row=oof[(str(seed),str(fold),rid)]
                for name in ['raw_et','raw_lgb','raw_mlp']:same(float(z[name][j]),float(row[name]))
            coverage.append(dict(fold=fold,seed=seed,train_days=len(keys),train_rows=len(ids),cache_rows=len(z['row_id'])))
    assert sets[0]==sets[1]==sets[2]
out=dict(status='PASS',checks=['state interaction','gap budget','equal fraction trim','unique-control match sensitivity','ordinary/high context','prefix counts','30 cache training-day full coverage and raw predictions'],gap_mse=gap,parts=parts,coverage=coverage)
with (H/'critic_verification_v2.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='coverage'},ensure_ascii=False,indent=2))
