"""Read existing public OOF labels and saved IDs only; no model run or new scoring."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
INNER=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
OLD=ROOT/'집/코덱스/local/statistical_experiments_20261003_v1'
ORIGINAL=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
labels={}
with (ORIGINAL/'v2_integration_oof.csv').open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        if row['validator']=='DIAG10' and row['seed']=='7':
            assert row['row_id'] not in labels
            labels[row['row_id']]=float(row['sub_ec'])
assert len(labels)==8640 and all(np.isfinite(list(labels.values())))

def ids(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in ['row_id','inner_train_id']}

def highdays(row_ids):
    groups={}
    for key in row_ids:
        farm,day,hour=str(key).split('_')
        groups.setdefault((farm,int(day)),[]).append(labels[str(key)])
    return sum(sum(y)/len(y)>=.8 for y in groups.values())

cells=[]
for path in sorted(INNER.glob('*_cpu.npz')):
    v,k=path.stem.rsplit('_',2)[:2]
    assert v in ['DIAG10','A','B','EXT10','EXT12']
    z=ids(path);old=ids(OLD/f'E_{v}_{k}_cpu.npz')
    assert len(set(z['row_id']))==len(z['row_id']) and len(set(z['inner_train_id']))==len(z['inner_train_id'])
    assert np.array_equal(z['row_id'],old['row_id']) and np.array_equal(z['inner_train_id'],old['inner_train_id'])
    assert set(z['row_id'])|set(z['inner_train_id']) <= set(labels)
    cells.append(dict(validator=v,fold=int(k),train_rows=len(z['inner_train_id']),query_rows=len(z['row_id']),inner_high_days=highdays(z['inner_train_id'])))

first=next(c for c in cells if c['validator']=='DIAG10' and c['fold']==0)
with np.load(ORIGINAL/'DIAG10_0_r3_7.npz',allow_pickle=False) as z:
    outer_high=highdays(z['train_row_id'])
record=dict(status='PASS',scope='Snapshot of completed public CPU ID caches only; no fit/prediction scoring',
            public_rows=len(labels),completed_cpu_caches=len(cells),cells=cells,
            firstfold_inner_high_days=first['inner_high_days'],firstfold_outer_high_days=outer_high,
            firstfold_fallback_executed_if_current_snapshot_used=(first['inner_high_days']<2 or outer_high<2))
(OUT/'nested_public_cache_check_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='cells'},ensure_ascii=False,indent=2))
