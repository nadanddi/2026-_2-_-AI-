"""Claude DPR/DPC/DCP1 public saved-output audit; no model import, fit or EL1 score.

Run from repository root with PYTHONPATH="" python -u <this file>.
EL1 rows are discarded while still strings, before any numeric label/prediction parse.
The colleague bootstrap is reproduced, not substituted for family20's preregistration.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env  # first project import; no training module is imported
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = ROOT / '집/클로드/research'
REF = ROOT / '집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
DEST = HERE / 'reload_verification_v6.json'
assert not DEST.exists(), 'Produced evidence must not be overwritten'
RESULT = {'status': 'PASS', 'checks': 0, 'scope': 'saved public output arithmetic only',
          'git_interval': ['283fc25', '8f13110'], 'experiments': {},
          'EL1_numeric_labels_or_predictions_parsed': 0,
          'fit': 0, 'predict': 0, 'test_reads': 0, 'raw_train_y_reads': 0}


def check(ok, detail):
    assert bool(ok), detail
    RESULT['checks'] += 1


def close(a, b, detail, tolerance=1e-12):
    check(abs(float(a) - float(b)) <= tolerance, (detail, a, b))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(rows):
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def public_reference():
    targets, layouts, actual_v2 = {}, defaultdict(dict), {}
    with REF.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['validator'] not in ('DIAG10', 'A', 'B', 'EXT10', 'EXT12'):
                continue
            key = row['row_id']
            target = float(row['sub_ec'])
            check(math.isfinite(target), ('reference finite', key))
            if key in targets:
                close(targets[key], target, ('reference repeated target', key))
            targets[key] = target
            if row['seed'] == '7':
                layout = layouts[row['validator']]
                layout_key = (int(row['validation_fold']), key)
                check(layout_key not in layout, ('reference unique', row['validator'], layout_key))
                layout[layout_key] = int(row['validation_fold'])
            if row['validator'] == 'DIAG10':
                actual_v2[(key, int(row['seed']))] = float(row['season_v2'])
    check(len(targets) == 8640, ('public targets', len(targets)))
    check(len(layouts['DIAG10']) == 8640, 'reference DIAG rows')
    return targets, layouts, actual_v2


def load_public(tag, allowed, seeds, baseline_prefix, candidate_prefix):
    path = SRC / 'local' / f'ec3_{tag}_all.csv'
    rows, skipped = [], Counter()
    fields = [f'{p}_{s}' for s in seeds for p in (baseline_prefix, candidate_prefix)]
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            validator = row['validator']
            if validator not in allowed:
                skipped[validator] += 1
                continue  # EL1 target and prediction strings are never converted
            out = {c: row[c] for c in ('row_id', 'farm', 'validator')}
            out.update({c: int(row[c]) for c in ('day', 'hour', 'validation_fold')})
            out.update({c: float(row[c]) for c in ['sub_ec'] + fields})
            check(all(math.isfinite(out[c]) for c in ['sub_ec'] + fields),
                  ('finite public numbers', tag, out['row_id']))
            check(out['row_id'] == f"{out['farm']}_{out['day']:03d}_{out['hour']:02d}",
                  ('row_id metadata', tag, out['row_id']))
            close(out['sub_ec'], TARGETS[out['row_id']], ('cached public label', tag, out['row_id']))
            rows.append(out)
    check(set(skipped).issubset({'EL1', 'A', 'B', 'EXT10', 'EXT12'}), ('excluded validators', tag, skipped))
    d = pd.DataFrame(rows)
    check(not d.duplicated(['validator', 'validation_fold', 'row_id']).any(), ('unique public rows', tag))
    check(set(d.validator) == set(allowed), ('validator inventory', tag))
    for v, g in d.groupby('validator'):
        for (fold, farm, day), h in g.groupby(['validation_fold', 'farm', 'day']):
            check(sorted(h.hour.tolist()) == list(range(24)), ('24h complete', tag, v, farm, day))
    return d, dict(skipped), canonical_sha(rows)


def score(g, baseline, candidate, label):
    y, a, b = g.sub_ec.to_numpy(), np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    ea, eb = a - y, b - y
    ra, rb = float(np.sqrt(np.mean(ea * ea))), float(np.sqrt(np.mean(eb * eb)))
    close(ra, math.sqrt(math.fsum(float(x) ** 2 for x in ea) / len(g)), (label, 'baseline fsum'))
    close(rb, math.sqrt(math.fsum(float(x) ** 2 for x in eb) / len(g)), (label, 'candidate fsum'))
    close(float(ea.mean()), math.fsum(float(x) for x in ea) / len(g), (label, 'baseline bias'))
    close(float(eb.mean()), math.fsum(float(x) for x in eb) / len(g), (label, 'candidate bias'))
    return dict(n=len(g), days=len(g[['farm', 'day']].drop_duplicates()), baseline=ra,
                candidate=rb, change_pct=100 * (rb / ra - 1),
                baseline_bias=float(ea.mean()), candidate_bias=float(eb.mean()))


def bootstrap(g, baseline, candidate, rng, label):
    a, b, y = np.asarray(baseline), np.asarray(candidate), g.sub_ec.to_numpy()
    delta = (b - y) ** 2 - (a - y) ** 2
    clusters = g.farm + '_' + (g.day // 5).astype(str)
    grouped = pd.Series(delta, index=g.index).groupby(clusters).agg(['sum', 'count'])
    manual = defaultdict(list)
    for key, value in zip(clusters, delta):
        manual[key].append(float(value))
    keys = sorted(manual)
    check(keys == grouped.index.tolist(), (label, 'sorted block keys'))
    ms = np.array([math.fsum(manual[k]) for k in keys])
    mc = np.array([len(manual[k]) for k in keys])
    for i, k in enumerate(keys):
        close(ms[i], grouped.loc[k, 'sum'], (label, k, 'block SSE'))
        close(mc[i], grouped.loc[k, 'count'], (label, k, 'block count'))
    ix = rng.integers(0, len(keys), (20000, len(keys)))
    ds = grouped['sum'].to_numpy()[ix].sum(axis=1) / grouped['count'].to_numpy()[ix].sum(axis=1)
    independent = ms[ix].sum(axis=1) / mc[ix].sum(axis=1)
    check(float(np.max(np.abs(ds - independent))) <= 1e-12, (label, 'all bootstrap arithmetic'))
    p = float((ds >= 0).mean())
    close(p, float((independent >= 0).mean()), (label, 'bootstrap sign fraction'))
    # A separate scalar fsum for a fixed spread of bootstrap draws also verifies resampling.
    for j in (0, 1, 7, 123, 999, 10007, 19999):
        scalar = math.fsum(float(ms[i]) for i in ix[j]) / sum(int(mc[i]) for i in ix[j])
        close(ds[j], scalar, (label, j, 'scalar resampling'))
    return dict(p_worse=p, draws=20000, blocks=len(keys),
                block_definition='farm + calendar record day//5; pooled, not farm-stratified',
                n=len(g), days=len(g[['farm', 'day']].drop_duplicates()),
                mse_delta=float(delta.mean()), descriptive_mse_delta_ci95=np.quantile(ds, [.025, .975]).tolist(),
                independent_max_delta=float(np.max(np.abs(ds - independent))))


def layout_check(tag, d, shift, validators):
    for v in validators:
        g = d[d.validator == v]
        if v in ('DIAG10', 'A', 'B'):
            expected = LAYOUTS[v]
            if tag != 'DPR':
                expected = {k: f for k, f in expected.items() if int(k[1].split('_')[1]) >= 179}
            check(set(zip(g.validation_fold, g.row_id)) == set(expected), (tag, v, 'reference fold-row inventory'))
            for row in g.itertuples():
                check(row.validation_fold == expected[(row.validation_fold, row.row_id)], (tag, v, 'reference fold', row.row_id))
        else:
            expected = set(TARGETS) if tag == 'DPR' else LATE_IDS
            check(set(g.row_id) == expected, (tag, v, 'shifted row inventory'))
            check(np.array_equal(g.validation_fold.to_numpy(), ((g.day.to_numpy() + shift) // 5) % 10),
                  (tag, v, 'new fold construction'))


def segment_scores(tag, d, seeds, prefix, main_validator):
    g = d[d.validator == main_validator]
    mean_daily = g.groupby(['farm', 'day']).sub_ec.transform('mean')
    selections = [('early', g.day < 179), ('late', g.day >= 179),
                  ('high_day_mean_ge1', mean_daily >= 1), ('ordinary_day_mean_lt1', mean_daily < 1)]
    result = []
    for name, mask in selections:
        h = g[mask]
        if h.empty:
            continue
        for seed in seeds:
            result.append(dict(segment=name, seed=seed,
                               **score(h, h[f'base_{seed}'], h[f'{prefix}_{seed}'], (tag, name, seed))))
    return result


def main():
    global TARGETS, LAYOUTS, ACTUAL_V2, LATE_IDS
    catalog = ROOT / '공용/데이터_단서_카탈로그.md'
    numbers = re.findall(r'^\| (6\.\d+) \|', catalog.read_text(encoding='utf-8'), flags=re.M)
    check(bool(numbers), 'catalog entries')
    RESULT['catalog_latest_before_execution'] = numbers[-1]
    RESULT['catalog_sha256_at_execution'] = sha(catalog)
    TARGETS, LAYOUTS, ACTUAL_V2 = public_reference()
    LATE_IDS = {k for k in TARGETS if int(k.split('_')[1]) >= 179}
    check(len(LATE_IDS) == 1104, 'same 46 public late days x24')
    RESULT['public_reference'] = dict(path=str(REF), sha256=sha(REF), rows=len(TARGETS),
                                    late_rows=len(LATE_IDS), late_days=46)
    specs = [('DPR', [11, 202, 3030], ['DIAG10s', 'A', 'B'], 'dp', 2, 'DIAG10s', 'ec3_DPR_dp1_replication_v1'),
             ('DPC', [31, 404, 5050], ['DIAG10', 'DIAG10t'], 'dp', 4, 'DIAG10t', 'ec3_DPC_dp1_confirmation_pass2_v1'),
             ('DCP1', [61, 707, 8080], ['DIAG10', 'DIAG10u'], 'cand', 1, 'DIAG10u', 'ec3_DCP1_season_twin_plus_operation_v1')]
    for tag, seeds, validators, prefix, shift, new_v, stem in specs:
        d, skipped, selected_sha = load_public(tag, validators, seeds, 'base', prefix)
        layout_check(tag, d, shift, validators)
        if tag != 'DPR':
            check(bool((d.day >= 179).all()), (tag, 'only late saved rows'))
        cells = []
        log = (SRC / f'{stem}.log').read_text(encoding='utf-8')
        for v in validators:
            g = d[d.validator == v]
            line = re.search(r'^\s*' + re.escape(v) + r'\s+.*$', log, re.M)
            check(line is not None, (tag, v, 'log score line'))
            for seed in seeds:
                cell = dict(validator=v, seed=seed,
                            **score(g, g[f'base_{seed}'], g[f'{prefix}_{seed}'], (tag, v, seed)))
                reported = re.search(r's' + str(seed) + r' ([.\d]+)->([.\d]+) \(([+\-.\d]+)%\)', line.group())
                check(reported is not None, (tag, v, seed, 'reported cell'))
                close(round(cell['baseline'], 4), float(reported[1]), (tag, v, seed, 'log baseline'))
                close(round(cell['candidate'], 4), float(reported[2]), (tag, v, seed, 'log candidate'))
                close(round(cell['change_pct'], 1), float(reported[3]), (tag, v, seed, 'log change'))
                cells.append(cell)
        g = d[d.validator == new_v]
        check(set(g[g.day >= 179].row_id) == LATE_IDS, (tag, 'same late label IDs'))
        boots = []
        if tag == 'DPR':
            rng = np.random.default_rng(20261004)
            for seed in seeds:
                for name, h in [('all', g), ('late', g[g.day >= 179])]:
                    boots.append(dict(seed=seed, scope=name,
                                      **bootstrap(h, h[f'base_{seed}'], h[f'{prefix}_{seed}'], rng, (tag, seed, name))))
            log_ps = re.search(r'DIAG10s P\(worse\) all: (\[[^\]]+\])  pass-2 only \(descriptive\): (\[[^\]]+\])', log)
            check(log_ps is not None, 'DPR bootstrap log')
            for name, values in [('all', json.loads(log_ps[1])), ('late', json.loads(log_ps[2]))]:
                for b, old in zip([b for b in boots if b['scope'] == name], values):
                    close(round(b['p_worse'], 4), old, (tag, name, 'log bootstrap'))
        else:
            bm = g[[f'base_{s}' for s in seeds]].mean(axis=1).to_numpy()
            cm = g[[f'{prefix}_{s}' for s in seeds]].mean(axis=1).to_numpy()
            # Independently average the three predictions, not the three seed RMSEs.
            f_bm = np.array([math.fsum(float(x) for x in row) / len(seeds) for row in g[[f'base_{s}' for s in seeds]].to_numpy()])
            f_cm = np.array([math.fsum(float(x) for x in row) / len(seeds) for row in g[[f'{prefix}_{s}' for s in seeds]].to_numpy()])
            check(float(np.max(np.abs(bm - f_bm))) < 1e-12, (tag, 'seed mean baseline'))
            check(float(np.max(np.abs(cm - f_cm))) < 1e-12, (tag, 'seed mean candidate'))
            boot = dict(scope='new_layout_seed_mean',
                        **bootstrap(g, bm, cm, np.random.default_rng(20261004), tag))
            boots.append(boot)
            mean_score = score(g, bm, cm, (tag, 'seed mean'))
            reported = re.search(re.escape(new_v) + r' pass-2 seed-mean RMSE ([.\d]+) -> ([.\d]+)  P\(worse\) ([.\d]+)', log)
            check(reported is not None, (tag, 'mean log'))
            close(round(mean_score['baseline'], 4), float(reported[1]), (tag, 'mean baseline log'))
            close(round(mean_score['candidate'], 4), float(reported[2]), (tag, 'mean candidate log'))
            close(round(boot['p_worse'], 4), float(reported[3]), (tag, 'mean bootstrap log'))
        check(f'{tag} decision: FAIL' in log, (tag, 'colleague rejection'))
        r = dict(seeds=seeds, public_rows=len(d), excluded_before_numeric_parse=skipped,
                 selected_public_sha256=selected_sha, cells=cells, bootstrap=boots,
                 segments=segment_scores(tag, d, seeds, prefix, new_v),
                 all_public_cells_improve=all(c['change_pct'] < 0 for c in cells),
                 colleague_decision='FAIL', independently_supported_decision='FAIL',
                 source_sha256=sha(SRC / f'{stem}.py'), log_sha256=sha(SRC / f'{stem}.log'),
                 EL1='log only; not rescored; 9/9 includes 3 unverified EL1 cells')
        if tag != 'DPR':
            check(boot['p_worse'] >= .025, (tag, 'failure independently sufficient without EL1 scoring'))
            r['new_layout_seed_mean_score'] = mean_score
        RESULT['experiments'][tag] = r

    # A paired, same-original-seed comparison substantiates R3S vs actual-v2 distinction.
    original_seeds = [7, 101, 2024]
    original, excluded, original_sha = load_public('DP1', ['DIAG10'], original_seeds, 'r3s', 'dp')
    baseline_comparison = []
    for name, mask in [('all', np.ones(len(original), dtype=bool)), ('late', original.day >= 179)]:
        g = original[mask]
        for seed in original_seeds:
            actual = np.array([ACTUAL_V2[(key, seed)] for key in g.row_id])
            baseline_comparison.append(dict(scope=name, seed=seed,
                **score(g, g[f'r3s_{seed}'], actual, ('R3S_vs_actual_season_v2', name, seed)),
                baseline_definition='colleague R3S (.6ET+.3LGB+.1MLP), after shrink/clip',
                candidate_definition='cached actual season_v2 (.8R3+.2PFN), after shrink/clip',
                candidate_is_new_model=False))
    RESULT['baseline_R3S_vs_actual_season_v2'] = baseline_comparison
    RESULT['original_DP1_selected_public_sha256'] = original_sha
    RESULT['original_DP1_excluded_before_numeric_parse'] = excluded
    RESULT['late_day_overlap'] = dict(unique_days=46, unique_rows=1104, repeated_days_across_new_layouts=46,
                                     new_target_days_added_by_seeds_or_layouts=0,
                                     label_ids_sha256=canonical_sha(sorted(LATE_IDS)))
    RESULT['source_hashes'] = {n: sha(SRC / n) for n in ['ec3_DP1_daily_operation_pattern_v1.py',
        'ec3_DC7_query_twin_season_v1.py', 'ec2_DC5_r3_season_v1.py', 'ec2_DC4_exact_twin_anchor_v1.py']}
    RESULT['limitations'] = [
        'No new fit, independent full feature replay, unseen holdout, train-bound replay or test evaluation.',
        'All late layouts reuse the same 46 public days. New seeds/layouts test calculation sensitivity, not new target evidence.',
        'Failure to reject alone cannot prove low power, true benefit, or statistical equivalence.',
        'DCP1 changes nine features and query-season rule together; different folds/seeds prevent a causal DP1-vs-DC7 attribution.',
        'DP1 fills masked/missing operations with zero; absence may be interpreted as closed/off.',
        'DCP1 DC7 requires hour>=5, but nanmean does not enforce six complete observed weather hours.',
        'Protocol6.279 is a read colleague procedure. Family20 strict15, p<.00125 and adjusted CI are untouched.',
        'Colleague EL1 figures are log provenance only; no EL1 label/prediction numeric conversion or rescoring.',
        'Public score arithmetic recheck is not evidence that current actual-v2 gains from appending these features.'
    ]
    check(not DEST.exists(), 'new output only')
    with DEST.open('x', encoding='utf-8') as stream:
        json.dump(RESULT, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': RESULT['status'], 'checks': RESULT['checks'],
                      'catalog_latest_before_execution': RESULT['catalog_latest_before_execution'],
                      'experiments': {k: {'public_rows': v['public_rows'], 'bootstrap': v['bootstrap'],
                                        'segments': v['segments']} for k, v in RESULT['experiments'].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
