"""Independent availability audit using only ID sets and scalar integer gaps."""
from pathlib import Path
import csv,json,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
with (HERE/'gap_audit.json').open(encoding='utf-8') as f:audit=json.load(f)
for target,path in [('TEMP',ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv'),('EC',ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv')]:
    with path.open(encoding='utf-8-sig',newline='') as f:
        keys={(r['farm'],int(float(r['day']))) for r in csv.DictReader(f)}
    # EC all validators cover precisely the public universe, not lock rows.
    with (ROOT/'공용/대회자료/정형데이터/test_X.csv').open(encoding='utf-8-sig',newline='') as f:
        ids=[r['row_id'] for r in csv.DictReader(f) if r['row_id'][:3] in ('F13','F47')]
    gaps=[]
    for row in ids:
        farm=row[:3];day=int(row[4:7]);past=[d for f,d in keys if f==farm and d<day and (d>=179)==(day>=179)];assert past;gaps.append(day-max(past))
    ordered=sorted(gaps);median=(ordered[(len(ordered)-1)//2]+ordered[len(ordered)//2])/2;pos=.9*(len(ordered)-1);lower=math.floor(pos);p90=ordered[lower]+(pos-lower)*(ordered[math.ceil(pos)]-ordered[lower])
    expected=next(r for r in audit['records'] if r['target']==target and r['validator']=='actual_metadata');assert len(ids)==expected['n_rows']==1440;assert len({r[:7] for r in ids})==expected['n_days']==60;assert median==expected['gap_median']==5;assert p90==expected['gap_p90']==10
result=dict(status='PASS',method='independent csv ID sets + integer subtraction + sorted quantile',targets=['TEMP','EC'],rows=1440,days=60,median_past_gap=5,p90_past_gap=10,test_targets_read=False,test_values_used=False)
(HERE/'gap_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
