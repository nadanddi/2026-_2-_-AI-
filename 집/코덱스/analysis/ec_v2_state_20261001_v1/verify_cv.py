from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,math
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=ROOT/'집/코덱스/local/ec_v2_state_20261001_v1'
assert json.loads((OUT/'cv_complete.json').read_text(encoding='utf-8'))['status']=='PASS'
d=pd.read_csv(OUT/'cv_predictions.csv');diag=d[d.validator.eq('DIAG10')].copy();assert len(diag)==8640 and diag.row_id.is_unique
locks={(z['farm'],int(z['day'])) for z in json.loads((Path(env.CODEX)/'ec_final_lock/locked_days.json').read_text(encoding='utf-8'))['selected']}
truth={}
with (Path(env.DATA)/'train_y.csv').open(newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        farm,day,_=r['row_id'].split('_')
        if farm not in ['F13','F47'] or (farm,int(day)) in locks:continue
        truth[r['row_id']]=float(r['sub_ec'])
assert set(diag.row_id)==set(truth)
for rid,y in zip(d.row_id,d.sub_ec):assert math.isclose(y,truth[rid],abs_tol=5e-15,rel_tol=0)
def rmse(y,p):
    a=float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)));b=math.sqrt(math.fsum((float(x)-float(v))**2 for x,v in zip(y,p))/len(y));assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-13);return a
rows=[];seeds=[]
for name,g in d.groupby('validator'):
    avg=g.groupby('row_id')[['sub_ec','baseline','candidate']].mean()
    for col in ['baseline','candidate']:rows.append({'validator':name,'model':col,'rmse':rmse(g.sub_ec,g[col]),'unique_average_rmse':rmse(avg.sub_ec,avg[col]),'row_occurrences':len(g),'unique_rows':len(avg)})
    for s in [7,101,2024]:
        b=rmse(g.sub_ec,g[f'baseline_{s}']);c=rmse(g.sub_ec,g[f'candidate_{s}']);seeds.append({'validator':name,'seed':s,'baseline_rmse':b,'candidate_rmse':c,'delta':c-b})
tables=[]
for farm,g in diag.groupby('farm'):
    a=pd.DataFrame({'block':g.block,'n':1,'baseline':(g.baseline-g.sub_ec)**2,'candidate':(g.candidate-g.sub_ec)**2});tables.append(a.groupby('block')[['n','baseline','candidate']].sum().to_numpy())
rng=np.random.default_rng(261041);tot=np.zeros((20000,3))
for a in tables:tot+=a[rng.integers(0,len(a),size=(20000,len(a)))].sum(axis=1)
delta=np.sqrt(tot[:,2]/tot[:,0])-np.sqrt(tot[:,1]/tot[:,0]);p=float(np.mean(delta>=0));ci=np.quantile(delta,[.025,.975]).tolist()
segments=[]
daily=diag.groupby(['farm','day']).sub_ec.mean();high=set(daily[daily.ge(1.2)].index);diag['high']=[(f,int(day)) in high for f,day in zip(diag.farm,diag.day)]
for name,mask in [('all',np.ones(len(diag),bool)),('F13',diag.farm.eq('F13')),('F47',diag.farm.eq('F47')),('early',diag.day.lt(179)),('late',diag.day.ge(179)),('high_ec',diag.high),('normal_ec',~diag.high)]:
    g=diag[mask]
    for col in ['baseline','candidate']:
        e=g[col]-g.sub_ec;level=e.groupby([g.farm,g.day]).transform('mean');ms=float((e**2).mean());ls=float((level**2).mean());shape=float(((e-level)**2).mean());assert math.isclose(ms,ls+shape,rel_tol=1e-11)
        segments.append({'segment':name,'model':col,'n':len(g),'rmse':rmse(g.sub_ec,g[col]),'bias':float(e.mean()),'level_fraction':ls/ms})
allcells=all(s['delta']<0 for s in seeds)
result={'status':'PASS','source_label_alignment':'PASS','diag_rows_once':8640,'independent_rmse_fsum':'PASS','segment_identity':'PASS','bootstrap_draws':20000,'difference_definition':'candidate minus baseline','difference_ci95':ci,'p_worse':p,'all_15_seed_validator_cells_improve':allcells,'numerical_screen_pass':bool(allcells and ci[1]<0 and p<.025),'candidate_adopted':False,'unused_confirmation_performed':False,'final_lock_scored':False,'temperature_model_created':False,'artifact_policy':'Generate user-requested experimental EC model even if screen fails; do not mark as adopted.'}
pd.DataFrame(rows).to_csv(OUT/'verified_scores.csv',index=False,encoding='utf-8-sig');pd.DataFrame(seeds).to_csv(OUT/'verified_seed_scores.csv',index=False,encoding='utf-8-sig');pd.DataFrame(segments).to_csv(OUT/'verified_segments.csv',index=False,encoding='utf-8-sig');(OUT/'cv_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(result,ensure_ascii=False,indent=2))
