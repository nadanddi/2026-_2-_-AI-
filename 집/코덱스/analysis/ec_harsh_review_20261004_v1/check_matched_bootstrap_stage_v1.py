"""Fixed public bootstrap and raw-OOF shift-stage replay; no model fit."""
from pathlib import Path
import sys, csv, json, math
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'집/클로드/research'))
import env
import numpy as np
H = ROOT/'집/코덱스/analysis/ec_matched_inner_calibration_20261003_v1'
SRC = ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
OUT = Path(__file__).resolve().parent
def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
report = json.loads((H/'full_verification_v1.json').read_text(encoding='utf-8'))
raw = table(SRC/'oof.csv'); cells = defaultdict(list)
for r in raw:
    cells[r['validator'],int(r['fold']),int(r['seed']),r['farm'],int(r['day'])].append(r)
boots = []
for seed in [7,101,2024]:
    rng = np.random.default_rng(20261003+seed)
    total = np.zeros(20000); count = np.zeros(20000)
    for farm in ['F13','F47']:
        keys = sorted((k for k in cells if k[0]=='DIAG10' and k[2]==seed and k[3]==farm), key=lambda k:k[4])
        assert len(keys) == len({k[4] for k in keys}) == 180
        sums = []; counts = []
        for i in range(0,len(keys),5):
            rows = [r for k in keys[i:i+5] for r in cells[k]]
            sums.append(math.fsum((float(r['candidate'])-float(r['y']))**2-(float(r['baseline'])-float(r['y']))**2 for r in rows))
            counts.append(len(rows))
        ix = rng.integers(len(sums),size=(20000,len(sums)))
        total += np.asarray(sums)[ix].sum(1)
        count += np.asarray(counts)[ix].sum(1)
    samples = total/count
    result = dict(seed=seed,p_worse=float(np.mean(samples>=0)),ci=np.quantile(samples,[report['alpha'],1-report['alpha']]).tolist())
    expected = report['bootstrap'][str(seed)]
    assert result['p_worse'] == expected['p_worse']
    assert max(abs(a-b) for a,b in zip(result['ci'],expected['ci'])) < 1e-12
    boots.append(result)

stage_maxdiff = 0.; stage_rows = 0
for r in table(SRC/'full_shift_records_v1.csv'):
    if r['domain'] != 'outer': continue
    key = r['validator'],int(r['fold']),int(r['seed']),r['farm'],int(r['day'])
    rows = cells[key]; h = int(r['hour']); assert len(rows)==24
    prefix = [z for z in rows if int(z['hour'])<=h]
    assert len(prefix)==h+1
    x = math.fsum(float(z['baseline']) for z in prefix)/len(prefix)
    yday = math.fsum(float(z['y']) for z in rows)/24
    current = [z for z in rows if int(z['hour'])==h]; assert len(current)==1
    correction = float(current[0]['correction'])
    for value,col in [(x,'x'),(yday,'yday'),(yday-x,'required_shift'),(correction,'learned_correction')]:
        stage_maxdiff = max(stage_maxdiff,abs(value-float(r[col])))
    stage_rows += 1
assert stage_maxdiff < 1e-12
record = dict(status='PASS',scope='Fixed completed public OOF only; reproduce registered farm-stratified 5-record bootstrap and outer-prefix diagnostic stage; no fit or tuning',
              bootstrap=boots,bootstrap_replicates=20000,alpha=report['alpha'],
              shift_outer_stage_rows=stage_rows,shift_outer_stage_maxdiff=stage_maxdiff)
path = OUT/'matched_bootstrap_stage_check_v1.json'; assert not path.exists()
path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
