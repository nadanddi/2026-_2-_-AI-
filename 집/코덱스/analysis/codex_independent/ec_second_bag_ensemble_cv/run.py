"""Exploratory exact-ensemble audit of already-scored second context bag."""
import importlib.util
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
BAG = HERE.parents[1] / 'tabpfn_cpu_bag'
SECOND = ROOT / 'local/tabpfn_cpu_second_bag/20260928_035601'
spec = importlib.util.spec_from_file_location('second_bag_ensemble_core', BAG / 'run.py')
bag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bag)
core, previous = bag.core, bag.previous
np, pd = bag.np, bag.pd
FOLDS = (0, 2, 4, 6)
SEEDS = (7, 101, 2024)


def score(frame, prediction):
    y = frame.sub_ec.to_numpy(float)
    p = frame[prediction].to_numpy(float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def resample(frame, seed):
    a = frame.copy()
    a['baseline_sq'] = (a.sub_ec - a.baseline) ** 2
    a['second_sq'] = (a.sub_ec - a.second) ** 2
    groups = {farm: g.groupby('block')[['baseline_sq', 'second_sq']].sum().to_numpy(float)
              for farm, g in a.groupby('farm')}
    rng = np.random.default_rng(seed)
    change = []
    for _ in range(5000):
        s = sum(t[rng.integers(0, len(t), len(t))].sum(axis=0)
                for t in groups.values())
        change.append(float(np.sqrt(s[1]/s[0]) - 1))
    return dict(block_counts={f: len(g) for f, g in groups.items()},
                ci95=[float(x) for x in np.quantile(change, [.025, .975])],
                share_improved=float(np.mean(np.asarray(change) < 0)), seed=seed)


def main():
    out = ROOT / 'local/ec_second_bag_ensemble_cv' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(code_hash=core.sha(HERE), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
                    second_result_hash=core.sha(SECOND / 'result.json'),
                    second_files={str(f): core.sha(SECOND / f'fold{f}_bag.csv') for f in FOLDS},
                    v1_hash=core.sha(HERE.parents[1] / 'rl_ec_v1/run.py'),
                    v2_hash=core.sha(HERE.parents[1] / 'rl_ec_v2/run.py'),
                    splits_hash=core.sha(bag.SPLITS), folds=FOLDS, seeds=SEEDS,
                    search_only=True, candidate_changed=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    dev = bag.prepare()
    scores = {}
    frames = []
    for fold in FOLDS:
        trm, vam = core.split(dev, fold)
        tr, va = dev[trm], dev[vam].reset_index(drop=True)
        saved = pd.read_csv(SECOND / f'fold{fold}_bag.csv')
        assert saved.row_id.tolist() == va.row_id.tolist()
        raw_members = []
        for seed in SEEDS:
            cache = ROOT / 'local/ec_baseline_cache' / (previous.cache_key(tr, va, seed) + '.npz')
            assert cache.is_file()
            with np.load(cache) as z:
                assert np.array_equal(z['row_id'], va.row_id.to_numpy(str))
                members = [z[n].copy() for n in ('et', 'lgb', 'mlp')]
            raw = .6 * members[0] + .3 * members[1] + .1 * members[2]
            baseline = np.clip(core.shrink(raw, va), tr.sub_ec.min(), tr.sub_ec.max())
            second = np.clip(core.shrink(.8 * raw + .2 * saved.second_bag.to_numpy(float), va),
                             tr.sub_ec.min(), tr.sub_ec.max())
            first = np.clip(core.shrink(.8 * raw + .2 * saved.first_bag.to_numpy(float), va),
                            tr.sub_ec.min(), tr.sub_ec.max())
            for got, name in ((baseline, f'baseline_{seed}'),
                              (second, f'second_{seed}'), (first, f'first_{seed}')):
                np.testing.assert_allclose(got, saved[name].to_numpy(float), rtol=0, atol=1e-12)
            raw_members.append(raw)
        raw = np.mean(raw_members, axis=0)
        row = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        row['baseline'] = np.clip(core.shrink(raw, va), tr.sub_ec.min(), tr.sub_ec.max())
        row['second'] = np.clip(core.shrink(.8 * raw + .2 * saved.second_bag.to_numpy(float), va),
                                tr.sub_ec.min(), tr.sub_ec.max())
        row['first'] = np.clip(core.shrink(.8 * raw + .2 * saved.first_bag.to_numpy(float), va),
                               tr.sub_ec.min(), tr.sub_ec.max())
        a, b, c = (score(row, col) for col in ('baseline', 'second', 'first'))
        scores[str(fold)] = dict(baseline=a, second=b, first=c,
                                 second_change=b/a-1, first_change=c/a-1)
        row.to_csv(out / f'fold{fold}.csv', index=False, float_format='%.17g')
        frames.append(row)
        print(f'FOLD {fold}: baseline={a:.6f} second={b:.6f} first={c:.6f}', flush=True)
    combined = pd.concat(frames, ignore_index=True)
    assert combined.row_id.is_unique
    a, b, c = (score(combined, col) for col in ('baseline', 'second', 'first'))
    by_farm = {}
    for farm, group in combined.groupby('farm'):
        x, y, z = (score(group, col) for col in ('baseline', 'second', 'first'))
        by_farm[farm] = dict(baseline=x, second=y, first=z,
                             second_change=y/x-1, first_change=z/x-1)
    day = combined.groupby(['farm', 'day']).apply(
        lambda g: pd.Series({'baseline': score(g, 'baseline'),
                             'second': score(g, 'second'),
                             'first': score(g, 'first')}),
        include_groups=False).reset_index()
    result = dict(scores=scores,
                  pooled=dict(baseline=a, second=b, first=c,
                              second_change=b/a-1, first_change=c/a-1),
                  by_farm=by_farm,
                  days=dict(total=len(day), second_improved=int((day.second < day.baseline).sum()),
                            first_improved=int((day['first'] < day.baseline).sum())),
                  bootstrap_five_day=resample(combined, 291001),
                  confirmation_scored=False, candidate_changed=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                     encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    with bag.threadpool_limits(limits=4):
        main()
