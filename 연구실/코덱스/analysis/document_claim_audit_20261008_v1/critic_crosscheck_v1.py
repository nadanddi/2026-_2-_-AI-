"""Independent stdlib-only arithmetic after common env bootstrap; no model fits."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import csv, json, math, statistics, hashlib
from collections import defaultdict

O = H / 'first_v4'
OLD = ROOT / '연구실/코덱스/analysis/ec_current14_influence_20261007_v1'
BASE = ROOT / '연구실/코덱스/local/ec_current14_influence_20261007_v1/baseline_rows.csv'
DATA = Path(env.DATA)
checks = 0
maxerr = 0.0

def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def eq(x, y):
    global checks
    checks += 1
    assert x == y, (x, y)

def near(x, y):
    global checks, maxerr
    checks += 1
    if y == '' or y is None:
        assert x is None or math.isnan(x)
        return
    x, y = float(x), float(y)
    if math.isnan(x) and math.isnan(y): return
    assert math.isfinite(x) and math.isfinite(y), (x, y)
    d = abs(x-y); maxerr = max(maxerr, d)
    assert d <= 2e-11 * max(1, abs(x), abs(y)), (x,y,d)

def corr(a, b):
    n = len(a)
    if n < 3: return None
    am, bm = math.fsum(a)/n, math.fsum(b)/n
    aa, bb = [v-am for v in a], [v-bm for v in b]
    va, vb = math.fsum(v*v for v in aa), math.fsum(v*v for v in bb)
    if va == 0 or vb == 0: return None
    return math.fsum(x*y for x,y in zip(aa,bb))/math.sqrt(va*vb)

def parse_bool(x):
    return None if x == '' else x == 'True'

m = json.loads((O/'manifest.json').read_text(encoding='utf-8'))
for name, digest in m['output_sha'].items(): eq(sha(O/name), digest)
for name, digest in m['data_sha'].items(): eq(sha(DATA/name), digest)
eq(sha(BASE), m['baseline_sha']); eq(sha(H/'audit_v4.py'), m['source_sha'])
prep = json.loads((OLD/'preparation_v4.json').read_text(encoding='utf-8'))
ids = [i for fold in prep['records'] for i in fold['query_ids']]
eq(len(ids),8640); eq(len(set(ids)),8640)
allowed = set(ids)
for fold in prep['records']:
    assert set(fold['train_ids']) <= allowed
    assert set(fold['train_ids']).isdisjoint(fold['query_ids'])

# Never convert a label value before the public-ID decision.
y = {}; skipped = 0
for r in rows(DATA/'train_y.csv'):
    if r['row_id'] not in allowed:
        skipped += 1
        continue
    assert r['row_id'] not in y
    y[r['row_id']] = float(r['sub_ec'])
eq(set(y),allowed); eq(skipped,m['skipped_label_rows_without_numeric_parse'])

raw = defaultdict(dict)
for group,name in [('train_inputs','train_X.csv'),('test_inputs','test_X.csv')]:
    for r in rows(DATA/name):
        f,d,h = r['row_id'].split('_')
        if f not in {'F13','F47'}: continue
        key=(group,f,int(d)); h=int(h)
        assert h not in raw[key]
        raw[key][h] = float(r['in_co2']) if r['in_co2'] else math.nan

day = {}
for r in rows(O/'day_inputs.csv'):
    key=(r['group'],r['farm'],int(r['day']))
    co=raw[key];eq(set(co),set(range(24)))
    dc=[co[h]-co[h-1] for h in range(1,24)]
    good=[v for v in dc if math.isfinite(v)]
    pairs=[(a,b) for a,b in zip(dc[:-1],dc[1:]) if math.isfinite(a) and math.isfinite(b)]
    med=statistics.median(map(abs,good)) if good else None
    ac=corr([p[0] for p in pairs],[p[1] for p in pairs]) if len(pairs)>=8 else None
    r1=med>14 if len(good)>=12 else None
    r2=(r1 and ac<0) if r1 is not None and ac is not None else None
    near(med,r['med']);near(ac,r['ac'])
    eq(r1,parse_bool(r['R1']));eq(r2,parse_bool(r['R2']))
    eq(len(good),int(r['valid_diffs']));eq(len(pairs),int(r['valid_ac_pairs']))
    eq(r['public']=='True',all(f"{r['farm']}_{int(r['day']):03d}_{h:02d}" in allowed for h in range(24)))
    day[key]=dict(R1=r1,R2=r2,source=r)
eq(len(day),460)

bucket=defaultdict(list)
for r in rows(BASE):
    assert r['row_id'] in allowed and r['arm']=='BASE'
    truth=y[r['row_id']];near(truth,r['sub_ec'])
    pred=float(r['prediction']);assert math.isfinite(pred)
    bucket[(r['seed'],r['farm'],int(r['day']))].append((r['row_id'],truth,pred))
daily={}
for key,v in bucket.items():
    eq(len(v),24);eq(len({z[0] for z in v}),24)
    truth=math.fsum(z[1] for z in v)/24;pred=math.fsum(z[2] for z in v)/24
    bias=math.fsum(z[2]-z[1] for z in v)/24;sse=math.fsum((z[2]-z[1])**2 for z in v)
    daily[key]=dict(truth=truth,prediction=pred,bias=bias,sse=sse,**day[('train_inputs',key[1],key[2])])
for r in rows(O/'ec_days.csv'):
    k=(r['seed'],r['farm'],int(r['day']));d=daily[k]
    for c in ['truth','prediction','bias','sse']: near(d[c],r[c])
eq(len(daily),1440)

def in_scope(k,d,s):
    if s=='all':return True
    if s in {'F13','F47'}:return k[1]==s
    if s=='pass1':return k[2]<179
    if s=='pass2':return k[2]>=179
    if s=='ordinary':return d['truth']<1
    if s=='high':return d['truth']>=1
    if s=='sealed':return d['source']['sealed']=='True'
    if s=='unsealed':return d['source']['sealed']=='False'
    if s=='pass2_ordinary':return k[2]>=179 and d['truth']<1
    raise ValueError(s)

for r in rows(O/'ec_rough.csv'):
    ds=[d for k,d in daily.items() if k[0]==r['seed'] and in_scope(k,d,r['scope'])]
    z=[d for d in ds if d[r['definition']] is True];o=[d for d in ds if d[r['definition']] is False]
    eq(len(ds),int(r['n']));eq(len(z),int(r['rough_n']));eq(len(o),int(r['other_n']))
    eq(len(ds)-len(z)-len(o),int(r['unknown_n']))
    mz=math.fsum(d['sse'] for d in z)/(24*len(z)) if z else math.nan
    mo=math.fsum(d['sse'] for d in o)/(24*len(o)) if o else math.nan
    for col,value in [('rough_mse',mz),('other_mse',mo),('rough_rmse',math.sqrt(mz)),('other_rmse',math.sqrt(mo)),('rmse_ratio',math.sqrt(mz/mo)),('mse_ratio',mz/mo),('rough_sse_share',math.fsum(d['sse'] for d in z)/math.fsum(d['sse'] for d in ds))]:near(value,r[col])

auc_verified=[]
for r in rows(O/'prediction_auc.csv'):
    ds=[d for k,d in daily.items() if k[0]==r['seed'] and in_scope(k,d,r['scope'])]
    cutoff=float(r['cutoff']);pos=[d['prediction'] for d in ds if d['bias']>cutoff];neg=[d['prediction'] for d in ds if d['bias']<=cutoff]
    # Exact pair comparison, independent of sklearn and rank implementations.
    value=math.fsum(1.0 if p>n else .5 if p==n else 0.0 for p in pos for n in neg)/(len(pos)*len(neg)) if pos and neg else math.nan
    eq(len(ds),int(r['n']));eq(len(pos),int(r['positive']));near(value,r['auc'])
    auc_verified.append(dict(seed=r['seed'],scope=r['scope'],cutoff=cutoff,auc=value))

out=dict(status='PASS_FIRST_STAGE_RAW_CO2_PUBLIC_LABELS_OOF_FSUM_PAIRWISE_AUC',checks=checks,max_absolute_error=maxerr,raw_days=460,public_labels=8640,skipped_label_rows_without_numeric_parse=skipped,baseline_rows=34560,daily_rows=1440,auc_rows=len(auc_verified),new_predictive_fits=0,source_sha=sha(__file__),first_manifest_sha=sha(O/'manifest.json'),limitations=['K1/K2 not independently recalculated here','No new split/generalization or treatment effect test','Original document sensor/real-house labels are unavailable'],auc=auc_verified)
with (H/'critic_crosscheck_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k not in ['auc','limitations']},ensure_ascii=False))
