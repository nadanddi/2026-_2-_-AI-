"""Read-only EC profiling; final-lock labels are skipped before numeric parsing."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집' / '클로드' / 'research'))
import env
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = Path(env.DATA)
LOCK = Path(env.CODEX) / 'ec_final_lock' / 'locked_days.json'
FARMS = ('F13', 'F47')

def keys(frame):
    k = frame.row_id.str.extract(r'^(F\d{2})_(\d{3})_(\d{2})$')
    assert k.notna().all().all()
    return frame.assign(farm=k[0], day=k[1].astype(int), hour=k[2].astype(int))

def summary(s):
    a = s.dropna()
    return {'n': int(a.size), 'missing': int(s.isna().sum()), 'unique': int(a.nunique()),
            'mean': float(a.mean()) if len(a) else None,
            'min': float(a.min()) if len(a) else None,
            'p01': float(a.quantile(.01)) if len(a) else None,
            'median': float(a.median()) if len(a) else None,
            'p99': float(a.quantile(.99)) if len(a) else None,
            'max': float(a.max()) if len(a) else None,
            'zeros': int(a.eq(0).sum()), 'negative': int(a.lt(0).sum())}

def compute():
    locks = {(z['farm'], int(z['day'])) for z in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    x = keys(pd.read_csv(DATA / 'train_X.csv'))
    # Only evaluation schema and missingness mask are accessed.
    mask = pd.read_csv(DATA / 'test_X.csv').drop(columns='row_id').isna().all()
    ec_rows, all_ids, skipped = [], [], 0
    with (DATA / 'train_y.csv').open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rid = row['row_id']; all_ids.append(rid)
            farm, day, hour = rid.split('_')
            if (farm, int(day)) in locks:
                skipped += 1
                continue
            if farm in FARMS and row['sub_ec'].strip():
                ec_rows.append((rid, float(row['sub_ec'])))
    y = pd.DataFrame(ec_rows, columns=['row_id', 'sub_ec'])
    d = x.merge(y, on='row_id', how='inner', validate='one_to_one')
    features = [c for c in mask.index if not bool(mask[c])]
    assert set(y.row_id).issubset(set(x.row_id))
    assert len(all_ids) == len(set(all_ids))
    assert not x.row_id.duplicated().any()
    assert not any((f, int(day)) in locks for f, day in zip(d.farm, d.day))
    assert d.hour.between(0,23).all()
    profiles = []
    for group, frame in [('all_train', x), ('target_train', x[x.farm.isin(FARMS)]), ('unlocked_ec', d)]:
        for c in mask.index:
            z = summary(frame[c]); z.update(group=group, column=c, dtype=str(frame[c].dtype))
            if c.endswith('_hum') or c.startswith('act_'):
                z['outside_nominal_range'] = int((frame[c].notna() & ~frame[c].between(0,100)).sum())
            elif c in ['out_rad', 'out_wspd', 'in_rad', 'in_co2']:
                z['outside_nominal_range'] = int(frame[c].lt(0).sum())
            else:
                z['outside_nominal_range'] = None
            profiles.append(z)
    pd.DataFrame(profiles).to_csv(OUT / 'column_profile.csv', index=False, encoding='utf-8-sig')
    coverage = []
    for farm, g in x.groupby('farm'):
        t = (g.day*24+g.hour).sort_values()
        counts = g.groupby('day').size()
        coverage.append({'farm':farm, 'rows':len(g), 'days':len(counts), 'day_min':int(g.day.min()),
                         'day_max':int(g.day.max()), 'non24h_days':int(counts.ne(24).sum()),
                         'gap_events':int(t.diff().gt(1).sum()),
                         'absent_hours_within_span':int(t.max()-t.min()+1-len(g)),
                         'observed_ec_rows':int(d.farm.eq(farm).sum())})
    pd.DataFrame(coverage).to_csv(OUT / 'farm_coverage.csv',index=False,encoding='utf-8-sig')
    m = d.sub_ec.mean()
    fm = d.groupby('farm').sub_ec.transform('mean')
    dm = d.groupby(['farm','day']).sub_ec.transform('mean')
    total = float(((d.sub_ec-m)**2).sum())
    parts = {'farm':float(((fm-m)**2).sum()), 'day_within_farm':float(((dm-fm)**2).sum()),
             'hour_within_day':float(((d.sub_ec-dm)**2).sum())}
    assert math.isclose(sum(parts.values()),total,rel_tol=1e-12)
    daily = d.groupby(['farm','day']).sub_ec.agg(['size','mean','min','max','std']).reset_index()
    daily['range'] = daily['max']-daily['min']
    daily.to_csv(OUT / 'unlocked_daily_ec.csv',index=False,encoding='utf-8-sig')
    by_farm = {f:{'ec':summary(g.sub_ec), 'days':int(g.day.nunique()),
                  'daily_range_median':float(daily.loc[daily.farm.eq(f),'range'].median()),
                  'constant_days':int(daily.loc[daily.farm.eq(f),'range'].eq(0).sum())}
               for f,g in d.groupby('farm')}
    period = d.assign(period=np.where(d.day.lt(179),'day<179','day>=179')).groupby(['farm','period']).sub_ec.agg(['size','mean','median','min','max'])
    period.to_csv(OUT / 'period_ec.csv',encoding='utf-8-sig')
    repeated = {}
    for f,g in d.groupby('farm'):
        p=g.sort_values(['day','hour']); adjacent=(p.day*24+p.hour).diff().eq(1)
        repeated[f] = {c:int((p[c].eq(p[c].shift()) & p[c].notna() & adjacent).sum()) for c in features+['sub_ec']}
    r={'source_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['train_X.csv','train_y.csv','test_X.csv']},
       'lock_sha256':hashlib.sha256(LOCK.read_bytes()).hexdigest(),
       'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__},
       'train_x_rows':len(x),'train_y_id_rows':len(all_ids),'train_farms':int(x.farm.nunique()),
       'target_train_rows':int(x.farm.isin(FARMS).sum()), 'locked_days_excluded':len(locks),'locked_label_rows_skipped':skipped,
       'ec_rows':len(d),'ec_days':len(daily),'usable_features':features,'masked_features':mask.index[mask].tolist(),
       'duplicate_ids':0,'unmatched_ec_ids':0,'duplicate_usable_vectors_extra':int(d[features].duplicated().sum()),
       'variance_ss':dict(total=total,**parts),'variance_fraction':{k:v/total for k,v in parts.items()},
       'by_farm':by_farm,'adjacent_equal_pairs':repeated,
       'cleaning_log':[{'action':'no value cleaning','before_target_rows':int(x.farm.isin(FARMS).sum()),
                       'after_target_rows':int(x.farm.isin(FARMS).sum())}],
       'statistical_tests':0,'model_training':False,'test_values_used_for_rules':False}
    return r

def cross_check(r):
    # Independent csv + Python arithmetic, no pandas groupby/variance functions.
    locks={(z['farm'],int(z['day'])) for z in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    groups=defaultdict(list); farms=defaultdict(list); values=[]
    with (DATA/'train_y.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            farm,day,_=row['row_id'].split('_')
            if farm not in FARMS or (farm,int(day)) in locks or not row['sub_ec'].strip(): continue
            v=float(row['sub_ec']); groups[(farm,int(day))].append(v); farms[farm].append(v); values.append(v)
    mean=math.fsum(values)/len(values)
    means={f:math.fsum(v)/len(v) for f,v in farms.items()}
    a=math.fsum(len(v)*(means[f]-mean)**2 for f,v in farms.items())
    b=math.fsum(len(v)*(math.fsum(v)/len(v)-means[k[0]])**2 for k,v in groups.items())
    c=math.fsum(math.fsum((t-math.fsum(v)/len(v))**2 for t in v) for v in groups.values())
    independent={'farm':a,'day_within_farm':b,'hour_within_day':c,'total':math.fsum((t-mean)**2 for t in values)}
    assert len(values)==r['ec_rows'] and len(groups)==r['ec_days']
    for k,v in independent.items(): assert math.isclose(v,r['variance_ss'][k],rel_tol=1e-11,abs_tol=1e-10), k
    for f,v in farms.items():
        z=r['by_farm'][f]['ec']; assert len(v)==z['n'] and min(v)==z['min'] and max(v)==z['max']
    return {'status':'PASS','rows':len(values),'days':len(groups),'variance_ss':independent,
            'checks':['EC rows/days','farm EC counts/min/max','four variance sums via csv/math']}

if __name__=='__main__':
    first=compute(); second=compute(); assert first==second
    second['verification']=cross_check(second)
    (OUT/'profile_result.json').write_text(json.dumps(second,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:second[k] for k in ['train_x_rows','train_y_id_rows','train_farms','ec_rows','ec_days','masked_features','variance_fraction','by_farm','verification']},ensure_ascii=False,indent=2))
