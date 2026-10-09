"""Critic independent read-only verification; no fitting or label scoring."""
from pathlib import Path
import csv, decimal, hashlib, json, math, sys, zipfile
H = Path(__file__).resolve().parent
P = json.loads((H / 'preparation_v1.json').read_text(encoding='utf-8'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(H.parents[3] / '.analysis-tools/python'))
import numpy as np
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
E = json.loads((H / 'execution_v1.json').read_text(encoding='utf-8'))
assert E['status'] == 'EXECUTION_COMPLETED_COMPARE_PENDING'
assert digest(P['zip']) == P['zip_sha']
temp = Path(P['temp_code']).parent
with zipfile.ZipFile(P['zip']) as arc:
    canonical = {}
    for n in arc.namelist():
        if '/온도/' in n and not n.endswith('/'):
            rel = n.split('/온도/', 1)[1]
            canonical[rel] = hashlib.sha256(arc.read(n)).hexdigest()
    assert canonical
    for rel, expected in canonical.items():
        assert digest(temp / rel) == expected, rel
for n, expected in P['data_sha'].items():
    assert digest(temp / 'data' / n) == expected
cache = Path(P['cache'])
assert digest(cache / 'tabpfn-v2-regressor.ckpt') == P['checkpoint_sha']
assert digest(H / 'run_v1.py') == E['wrapper_sha']
assert digest(temp / 'code/make_submission_v13_temp.py') == E['source_sha']
assert all(Path(p).resolve().is_relative_to(Path(P['temp_code']).resolve()) for p in E['project_module_paths'].values())
manifest = json.loads((temp / 'config/manifest_12.json').read_text(encoding='utf-8'))
assert P['data_sha'] == manifest['inputs']
assert P['checkpoint_sha'] == manifest['tabpfn']['checkpoint']['sha256']
for n, expected in manifest['season_index']['sha256'].items():
    assert digest(temp / 'code/season_ec' / n) == expected
out = Path(P['output'])
generated = rows(out / 'temp_candidate_v13.csv')
reference = rows(temp / 'temp_candidate_v13.csv')
submitted_path = H.parents[3] / '제출/09회차_2026-10-06(팀)/submission_14.csv'
assert digest(submitted_path) == P['submission_sha']
submitted = rows(submitted_path)
ids = [r['row_id'] for r in generated]
assert len(ids) == len(set(ids)) == 1440
assert ids == [r['row_id'] for r in reference] == [r['row_id'] for r in submitted]
dec = decimal.Decimal
assert all(dec(a['sub_temp']) == dec(b['sub_temp']) for a, b in zip(reference, submitted))
diff = [float(a['sub_temp']) - float(b['sub_temp']) for a,b in zip(generated, submitted)]
assert all(math.isfinite(float(r['sub_temp'])) for r in generated)
changed = sum(dec(a['sub_temp']) != dec(b['sub_temp']) for a,b in zip(generated, submitted))
zpath = out / 'temp_candidate_v13_members.npz'
assert digest(zpath) == E['output_files'][zpath.name]
with np.load(zpath, allow_pickle=True) as z:
    assert z['row_id'].tolist() == ids
    base, codex, pf, gate, pred = [z[n].copy() for n in ['base','codex','pfn_samples','gate','pred']]
assert pf.shape == (8,1440)
assert all(a.shape == (1440,) for a in [base,codex,gate,pred])
assert all(np.isfinite(a).all() for a in [base,codex,pf,gate,pred])
test = {r['row_id']: float(r['in_temp']) if r['in_temp'] else math.nan for r in rows(temp / 'data/test_X.csv')}
max_scalar = 0.0
for i, rid in enumerate(ids):
    t = test[rid]
    g = 1.0 if math.isnan(t) else min(1.0,max(0.0,(t-8.0)/2.0))
    assert g == float(gate[i])
    avg = math.fsum(float(pf[j,i]) for j in range(8)) / 8.0
    scalar = math.fsum([0.4*float(base[i]),(0.2+0.4*(1.0-g))*float(codex[i]),0.4*g*avg])
    max_scalar = max(max_scalar, abs(scalar-float(pred[i])))
    assert dec(format(float(pred[i]), '.6f')) == dec(generated[i]['sub_temp'])
assert max_scalar < 1e-10
log = (H / 'run_v1.log').read_text(encoding='utf-8')
assert 'base done' in log and 'codex done' in log
assert all('tabpfn sample %d done' % i in log for i in range(1,9))
assert 'EXECUTION_COMPLETED' in log
assert 'train rows 9600 (down-weighted 2341) | test rows 1440 | base cols 135 | codex cols 89' in log
R = dict(status='PASS_EXACT_TEMPERATURE_6_DECIMALS' if changed == 0 else 'EXECUTED_NOT_EXACT',rows=1440,changed_rows=changed,
    max_abs_csv_diff=max(map(abs,diff)),rms_csv_diff=math.sqrt(math.fsum(d*d for d in diff)/1440),scalar_fsum_max_abs=max_scalar,
    gate_and_npz_ids_verified=True,zip_canonical_files_verified=len(canonical),source_and_checkpoint_preserved=True,complete_member_logs=True,
    generated_equals_zip_reference_bytes=digest(out/'temp_candidate_v13.csv')==digest(temp/'temp_candidate_v13.csv'),
    scope='온도만 재현. 출력 EC는 submission04 복사. 평가정답/리더보드 채점 및 추가 temp_v13_checks 실행 없음.',
    python=E['python'],python_reference=manifest['python'],versions=E['versions'],verification_sha=digest(__file__))
with (H/'critic_verification_v1.json').open('x',encoding='utf-8') as f:
    json.dump(R,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(R,ensure_ascii=False,indent=2))
