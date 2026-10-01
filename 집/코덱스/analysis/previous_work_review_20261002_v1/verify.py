from pathlib import Path
import csv, json, math, hashlib, sys
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'집/클로드/research'))
import env
import numpy as np

HERE = Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
locks = {(r['farm'], int(r['day'])) for r in json.loads((Path(env.CODEX)/'ec_final_lock/locked_days.json').read_text(encoding='utf-8'))['selected']}
truth = {}
for r in read(Path(env.DATA)/'train_y.csv'):
    farm, day, _ = r['row_id'].split('_')
    if farm in ['F13','F47'] and (farm, int(day)) not in locks:
        truth[r['row_id']] = float(r['sub_ec'])
oof = read(ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1/oof_predictions.csv')
cv = read(ROOT/'집/코덱스/local/ec_v2_state_20261001_v1/cv_predictions.csv')
scores = {}
for label, rows, cols in [('v2', oof, ['v2']), ('state', cv, ['baseline','candidate'])]:
    grouped = defaultdict(list)
    for r in rows:
        assert math.isclose(float(r['sub_ec']), truth[r['row_id']], abs_tol=5e-15)
        grouped[r['validator']].append(r)
    for v, g in grouped.items():
        for col in cols:
            e = [float(r[col])-float(r['sub_ec']) for r in g]
            a = math.sqrt(math.fsum(x*x for x in e)/len(e))
            b = float(np.sqrt(np.mean(np.square(e))))
            assert math.isclose(a,b,abs_tol=1e-12)
            scores[f'{label}/{v}/{col}'] = a
diag = [r for r in oof if r['validator']=='DIAG10']
assert len(diag)==8640 and len({r['row_id'] for r in diag})==8640
days = defaultdict(list)
for r in diag: days[(r['farm'],int(r['day']))].append(r)
ss = math.fsum((float(r['v2'])-float(r['sub_ec']))**2 for r in diag)
level = math.fsum(len(g)*(math.fsum(float(r['v2'])-float(r['sub_ec']) for r in g)/len(g))**2 for g in days.values())
shape = math.fsum(math.fsum(((float(r['v2'])-float(r['sub_ec']))-math.fsum(float(z['v2'])-float(z['sub_ec']) for z in g)/len(g))**2 for r in g) for g in days.values())
assert math.isclose(ss,level+shape,rel_tol=1e-12)
inputs = {r['row_id']:r for r in read(Path(env.DATA)/'train_X.csv') if r['row_id'] in truth}
sealed_high = []
for key,g in days.items():
    if math.fsum(float(inputs[r['row_id']]['act_circfan']) for r in g)/len(g)<10 and sum(float(inputs[r['row_id']]['act_vent'])==0 for r in g)/len(g)>.85 and math.fsum(float(r['sub_ec']) for r in g)/len(g)>=1.2:
        sealed_high.extend(g)
share = math.fsum((float(r['v2'])-float(r['sub_ec']))**2 for r in sealed_high)/ss
artifact = ROOT/'집/코덱스/local/ec_v2_state_20261001_v1/artifact_numpy253'
manifest = json.loads((artifact/'delivery_manifest.json').read_text(encoding='utf-8'))
hashes = {name: hashlib.sha256((artifact/name).read_bytes()).hexdigest()==expected for name,expected in manifest['files_sha256'].items()}
assert all(hashes.values())
pred = read(artifact/'ec_v2_state_v1_numpy253.csv')
submitted = read(ROOT/'제출/07회차_2026-10-01(팀)/submission_09.csv')
pmap = {r['row_id']:float(r['sub_ec']) for r in pred}
assert len(pmap)==1440 and len(submitted)==1440 and set(pmap)=={r['row_id'] for r in submitted}
diff = max(abs(float(r['sub_ec'])-pmap[r['row_id']]) for r in submitted)
result = {'status':'PASS','scores':scores,'diag_days':len(days),'diag_rows':len(diag),'level_error_share':level/ss,'shape_oracle_rmse':math.sqrt(shape/len(diag)),'sealed_high_days':len(sealed_high)//24,'sealed_high_error_share':share,'artifact_hash_matches':hashes,'submitted_state_max_difference':diff,'model_retrained':False,'final_lock_scored':False}
(HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
