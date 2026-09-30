from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import csv,json,math,hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=ROOT/'집'/'코덱스'/'local'/'ec_restart_phase3_20261001_v1'
COLS=['mean','farm_mean','ridge','raw_et','full_et','r3','v2']

def rmse(y,p):return float(np.sqrt(np.mean((np.array(y)-np.array(p))**2)))
def main():
    completed=json.loads((HERE/'completion.json').read_text(encoding='utf-8'))
    assert completed['status']=='PASS'
    d=pd.read_csv(OUT/'oof_predictions.csv')
    diag=d[d.validator.eq('DIAG10')].copy()
    assert len(diag)==8640 and diag.row_id.is_unique and len(diag[['farm','day']].drop_duplicates())==360
    # Read labels independently and skip lock values before conversion.
    locks={(z['farm'],int(z['day'])) for z in json.loads((Path(env.CODEX)/'ec_final_lock'/'locked_days.json').read_text(encoding='utf-8'))['selected']}
    truth={}
    with (Path(env.DATA)/'train_y.csv').open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            farm,day,_=r['row_id'].split('_')
            if farm not in ['F13','F47'] or (farm,int(day)) in locks:continue
            truth[r['row_id']]=float(r['sub_ec'])
    assert set(diag.row_id)==set(truth)
    for rid,y in zip(d.row_id,d.sub_ec):assert math.isclose(y,truth[rid],rel_tol=0,abs_tol=5e-15)
    fold=pd.read_csv(HERE/'fold_scores.csv')
    rows=[];seg=[]
    for validator,g in d.groupby('validator'):
        avg=g.groupby('row_id',sort=True)[['sub_ec']+COLS].mean()
        for c in COLS:
            f=fold[fold.validator.eq(validator)&fold.model.eq(c)]
            value=rmse(g.sub_ec,g[c])
            independent=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.sub_ec,g[c]))/len(g))
            assert math.isclose(value,independent,rel_tol=1e-12,abs_tol=1e-13)
            rows.append({'validator':validator,'model':c,'rmse_occurrence':value,'rmse_unique_average':rmse(avg.sub_ec,avg[c]),
                         'row_occurrences':len(g),'unique_rows':len(avg),'overlap_occurrences':len(g)-len(avg),
                         'fold_rmse_mean':float(f.rmse.mean()),'fold_rmse_std':float(f.rmse.std(ddof=0))})
    daily_mean=diag.groupby(['farm','day']).sub_ec.mean()
    diag['high_ec']=[daily_mean.loc[(f,int(day))]>=1.2 for f,day in zip(diag.farm,diag.day)]
    for name,mask in [('all',np.ones(len(diag),bool)),('F13',diag.farm.eq('F13')),('F47',diag.farm.eq('F47')),
                      ('early',diag.day.lt(179)),('late',diag.day.ge(179)),('hour0',diag.hour.eq(0)),('hour6',diag.hour.eq(6)),
                      ('high_ec',diag.high_ec),('normal_ec',~diag.high_ec)]:
        g=diag[mask]
        for c in COLS:
            er=g[c]-g.sub_ec
            dayerr=er.groupby([g.farm,g.day]).transform('mean')
            mse=float(np.mean(er**2));level=float(np.mean(dayerr**2));shape=float(np.mean((er-dayerr)**2))
            assert math.isclose(mse,level+shape,rel_tol=1e-11,abs_tol=1e-12)
            seg.append({'segment':name,'model':c,'n':len(g),'days':len(g[['farm','day']].drop_duplicates()),'rmse':math.sqrt(mse),
                        'bias':float(er.mean()),'level_error_fraction':level/mse,'level_rmse':math.sqrt(level),'shape_rmse':math.sqrt(shape)})
    # DIAG10 paired block bootstrap, preserving farm strata and all hours.
    tables={}
    for f,g in diag.groupby('farm'):
        a=pd.DataFrame({'block':g.block,'n':1,'v2':(g.v2-g.sub_ec)**2})
        for c in ['ridge','raw_et','full_et']:a[c]=(g[c]-g.sub_ec)**2
        tables[f]=a.groupby('block')[['n','v2','ridge','raw_et','full_et']].sum().to_numpy(float)
    rng=np.random.default_rng(261003);sums=np.zeros((20000,5))
    for a in tables.values():
        draws=rng.integers(0,len(a),size=(20000,len(a)))
        sums+=a[draws].sum(axis=1)
    base=np.sqrt(sums[:,1]/sums[:,0]);boot={}
    for j,c in enumerate(['ridge','raw_et','full_et'],2):
        delta=np.sqrt(sums[:,j]/sums[:,0])-base
        boot[c]={'draws':20000,'difference_rmse':rmse(diag.sub_ec,diag[c])-rmse(diag.sub_ec,diag.v2),
                 'ci_bonferroni3':np.quantile(delta,[.025/3,1-.025/3]).tolist(),'p_worse':float(np.mean(delta>=0)),
                 'improving_diagonal_folds':int((fold.query("validator=='DIAG10' and model==@c").set_index('fold').rmse < fold.query("validator=='DIAG10' and model=='v2'").set_index('fold').rmse).sum())}
    seed_report=[]
    for vn,g in d.groupby('validator'):
        for s in [7,101,2024]:
            for c in ['raw_et','full_et','r3']:
                a=rmse(g.sub_ec,g[f'{c}_{s}']);b=rmse(g.sub_ec,g[f'v2_{s}'])
                seed_report.append({'validator':vn,'seed':s,'model':c,'rmse':a,'v2_rmse':b,'relative_change':a/b-1})
    pd.DataFrame(rows).to_csv(HERE/'summary_scores.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(seg).to_csv(HERE/'segment_scores.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(seed_report).to_csv(HERE/'seed_comparisons.csv',index=False,encoding='utf-8-sig')
    checks={'source_labels_all_occurrences':'PASS','diag_every_unlocked_day_once':'PASS','all_aggregate_rmse_math_fsum':'PASS',
            'segment_error_decomposition':'PASS','raw_prediction_hash':hashlib.sha256((OUT/'oof_predictions.csv').read_bytes()).hexdigest()}
    for p in OUT.glob('*.npz'):
        m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(p.read_bytes()).hexdigest()==m['prediction_sha256']
    checks['checkpoint_hashes']='PASS'
    result={'verification':checks,'bootstrap_DIAG10':boot,'candidate_adopted':False,'final_lock_scored':False,
            'interpretation':'exploratory benchmark; published leaderboard cannot be used to map these RMSEs to hidden evaluation'}
    (HERE/'verified_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).pivot(index='model',columns='validator',values='rmse_occurrence').to_string())
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
