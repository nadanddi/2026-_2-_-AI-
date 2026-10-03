"""Stored public OOF and submission-package audit; no fitting or reserved labels."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import hashlib, io, json, math, re, zipfile
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
checks = []
def check(name, ok, **details):
    checks.append(dict(name=name, passed=bool(ok), **details))
    if not ok:
        raise AssertionError((name, details))

folder = ROOT / '제출/08회차_2026-10-03(팀)'
submission = (folder / 'submission_11.csv').read_bytes()
check('submission_sha', sha(submission) == 'ab7ab734e5a3016a8f7679b9a96cca6597de493eff490134f0f6c72e6c98c17e')
archive = next(folder.glob('*.zip'))
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    def read_suffix(s):
        matches = [n for n in names if n.endswith(s)]
        check('unique_zip_suffix:' + s, len(matches) == 1)
        return z.read(matches[0])
    check('zip_submission_bytes', read_suffix('/submission_11.csv') == submission)
    temp_bytes = read_suffix('/temp_candidate_v12.csv')
    ec_bytes = read_suffix('/EC/submission_10.csv')
    tm = json.loads(read_suffix('/config/manifest_12.json'))
    em = json.loads(read_suffix('/EC/manifest.json'))
    check('temp_component_hash', sha(temp_bytes) == tm['output_sha256'])
    check('ec_component_hash', sha(ec_bytes) == em['csv_sha256'])
    doc = next(folder.glob('설명자료*.md')).read_text(encoding='utf-8')
    # Source document includes one introductory preservation comment.
    root_readme = next(n for n in names if n.count('/') == 1 and n.endswith('/README.md'))
    check('explanation_readme_copy', doc.split('\n', 1)[1].strip() == z.read(root_readme).decode('utf-8-sig').replace('\r\n', '\n').strip())

frames = [pd.read_csv(io.BytesIO(b), float_precision='round_trip') for b in (submission, temp_bytes, ec_bytes)]
S, T, E = frames
for name, f in zip(('submitted', 'temperature', 'EC'), frames):
    check(name + '_rows_ids', len(f) == 1440 and f.row_id.is_unique and set(f.row_id) == set(S.row_id))
check('finite_submission', bool(np.isfinite(S[['sub_temp', 'sub_ec']].to_numpy()).all()))
merge_diffs = {}
for col, f in (('sub_temp', T), ('sub_ec', E)):
    merged = S[['row_id', col]].merge(f[['row_id', col]], on='row_id', validate='one_to_one', suffixes=('_submitted', '_component'))
    diff = float(np.max(np.abs(merged[col+'_submitted'] - merged[col+'_component'])))
    check('merge:' + col, diff <= 5.00001e-7, max_abs_diff=diff)
    merge_diffs[col] = diff

experiments = {
    'TS1': ('ec3_TS1_all.csv', 'ec3_TS1_two_stage_daymean_v1.log', 'ts'),
    'LG1': ('ec3_LG1_all.csv', 'ec3_LG1_hour_lag_inputs_v1.log', 'lg'),
    'LR1': ('ec3_LR1_all.csv', 'ec3_LR1_log_ratio_target_v1.log', 'lr'),
    'DI1': ('ec3_DI1_all.csv', 'ec3_DI1_drop_h0_fingerprint_v1.log', 'di'),
    'MW1': ('ec3_MW1_all.csv', 'ec3_MW1_multiday_weather_v1.log', 'mw'),
}
rows = []
keys = ['validator', 'validation_fold', 'row_id']
reference = None
for exp, (csv, log, prefix) in experiments.items():
    p = ROOT / '집/클로드/research/local' / csv
    f = pd.read_csv(p, float_precision='round_trip')
    check(exp + '_unique_public_rows', not f.duplicated(keys).any())
    indexed = f.set_index(keys).sort_index()
    if reference is None:
        reference = indexed[['sub_ec', 'r3s_7', 'r3s_101', 'r3s_2024']]
    else:
        check(exp + '_same_labels_baseline', indexed[reference.columns].equals(reference))
    text_log = (ROOT / '집/클로드/research' / log).read_text(encoding='utf-8')
    for v in ('DIAG10', 'A', 'B', 'EXT10', 'EXT12', 'EL1'):
        g = f.loc[f.validator == v]
        line = next(l for l in text_log.splitlines() if re.match(r'^\s*' + v + r'\s+s7\s', l))
        logged = {int(s): (float(a), float(b)) for s, a, b in re.findall(r's(\d+)\s+([\d.]+)->([\d.]+)', line)}
        for s in (7, 101, 2024):
            vals = []
            for col in ('r3s_' + str(s), prefix + '_' + str(s)):
                err = g[col].to_numpy() - g.sub_ec.to_numpy()
                a = float(np.sqrt(np.mean(err**2)))
                b = math.sqrt(math.fsum(float(e)**2 for e in err) / len(err))
                check(exp + '_' + v + '_' + col + '_independent_rmse', abs(a-b) < 1e-12)
                vals.append(a)
            check(exp + '_' + v + '_' + str(s) + '_log_rounding', all(abs(a-b) <= .000050001 for a,b in zip(vals, logged[s])))
            rows.append(dict(experiment=exp, validator=v, seed=s, rows=len(g), days=len(g[['farm','day']].drop_duplicates()), baseline_rmse=vals[0], candidate_rmse=vals[1], change_percent=100*(vals[1]/vals[0]-1)))
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / 'independent_scores_v1.csv', index=False)
for exp in experiments:
    q = metrics.loc[metrics.experiment == exp]
    check(exp + '_all_direction_rule_fails', bool((q.change_percent >= 0).any()))

inventory = []
for folder_name in ('statistical_experiments_20261003_v1', 'temp_tk_season_20261003_v1', 'two_days_full_20261003_v1'):
    base = ROOT / '집/코덱스/analysis' / folder_name
    for p in sorted(base.iterdir()):
        if p.suffix in ('.log', '.png'):
            data = p.read_bytes()
            inventory.append(dict(path=str(p.relative_to(ROOT)), size=len(data), sha256=sha(data)))
pd.DataFrame(inventory).to_csv(OUT / 'saved_extra_artifacts_v1.csv', index=False)
result = dict(status='PASS', checks=checks, metric_cells=len(metrics), saved_extra_files=len(inventory),
    submission_sha256=sha(submission), archive_sha256=sha(archive.read_bytes()), merge_max_diff=merge_diffs,
    temp_formula=tm['formula'], EC_recipe=em['recipe'],
    limitations=['No model retraining or full package reproduction', 'No reserved labels or lock files read',
        'Stored public OOF only; bootstrap p-values compared to logs, not recomputed',
        'Local literature notes reviewed; original papers not freshly verified'])
(OUT / 'verification_v1.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','metric_cells','saved_extra_files','archive_sha256','merge_max_diff')}, ensure_ascii=False))
print(metrics.groupby('experiment').change_percent.agg(['min','max']).to_string())
