from pathlib import Path
import sys,csv,json,math
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1'
diag=pd.read_csv(OUT/'oof_predictions.csv')
diag=diag[diag.validator.eq('DIAG10')]
daily=diag.groupby(['farm','day']).sub_ec.mean()
mask=pd.Series([(f,int(d)) in set(daily[daily.ge(1.2)].index) for f,d in zip(diag.farm,diag.day)],index=diag.index)
fraction=float(((diag.loc[mask,'v2']-diag.loc[mask,'sub_ec'])**2).sum()/((diag.v2-diag.sub_ec)**2).sum())
records=[]
with (OUT/'oof_predictions.csv').open(newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if r['validator']=='DIAG10':records.append(r)
total=math.fsum((float(r['v2'])-float(r['sub_ec']))**2 for r in records)
high=math.fsum((float(r['v2'])-float(r['sub_ec']))**2 for r in records if daily.loc[(r['farm'],int(r['day']))]>=1.2)
assert math.isclose(fraction,high/total,rel_tol=1e-12)
fold=pd.read_csv(HERE/'fold_scores.csv')
tr=[]
for name,g in fold[fold.validator.eq('DIAG10')&fold.train_rmse.notna()].groupby('model'):
    tr.append({'model':name,'training_rmse_fold_mean':float(g.train_rmse.mean()),'validation_rmse_fold_mean':float(g.rmse.mean())})
seeds=pd.read_csv(HERE/'seed_comparisons.csv')
direction=seeds.groupby('model').relative_change.agg(['min','max',lambda a:int(a.lt(0).sum()),'size']).reset_index()
test=pd.read_csv(Path(env.DATA)/'test_X.csv',usecols=['row_id'])
day=test.row_id.str[4:7].astype(int)
result={'status':'PASS','high_ec_days':int(daily.ge(1.2).sum()),'high_ec_error_fraction':fraction,'independent_csv_fsum':'PASS','training_validation_fold_means':tr,'baseline_beats_v2_seed_validator_cells':direction.to_dict(orient='records'),'evaluation_ID_only':{'rows':len(test),'day_min':int(day.min()),'day_max':int(day.max()),'day_ge179_rows':int(day.ge(179).sum())},'note':'Training means are in-sample scores, not independent validation. No complete v2 train score computed.'}
(HERE/'diagnostics_verified_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
