"""Feature-level rule checks before expensive four-context inference."""
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
V1 = HERE.parents[1] / 'rl_ec_v1'
sys.path.insert(0, str(V1))
import env  # noqa: E402
spec = importlib.util.spec_from_file_location('ec_core', V1 / 'run.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
np, pd = core.np, core.pd


def same(a, b):
    return bool(np.array_equal(a, b, equal_nan=True))


def main():
    out = ROOT / 'local/tabpfn_ec_end_to_end' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(env.DATA / 'train_X.csv')
    test = pd.read_csv(env.DATA / 'test_X.csv')
    assert len(test) == 1440 and test.row_id.is_unique
    train = train[train.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    assert set(test.row_id.str[:3]) == {'F13', 'F47'}
    a, b = core.identify(train), core.identify(test)
    assert not set(zip(a.farm, a.day)) & set(zip(b.farm, b.day)), 'Same-day train/test overlap'
    assert (b.groupby(['farm', 'day']).size() == 24).all(), 'Incomplete test day'
    assert b.row_id.is_unique
    tr_features = core.features(train).set_index('row_id')[core.FULL]
    te_features = core.features(test).set_index('row_id')[core.FULL]
    shuffled = core.features(test.sample(frac=1, random_state=287)).set_index('row_id')[core.FULL]
    assert same(te_features.to_numpy(), shuffled.loc[te_features.index].to_numpy())
    combined = core.features(pd.concat([train, test], ignore_index=True)).set_index('row_id')[core.FULL]
    assert same(te_features.to_numpy(), combined.loc[te_features.index].to_numpy())
    assert same(tr_features.to_numpy(), combined.loc[tr_features.index].to_numpy())

    records = []
    for farm in ('F13', 'F47'):
        f = b.farm.eq(farm)
        times = b.day * 24 + b.hour
        for frac in (.1, .4, .7, .9):
            cut = int(times[f].quantile(frac))
            past = f & times.le(cut)
            future = f & times.gt(cut)
            changed = test.copy()
            rng = np.random.default_rng(88000 + cut)
            for col in core.RAW:
                values = changed.loc[future, col].to_numpy(float)
                changed.loc[future, col] = values * 13 + rng.normal(400, 10, len(values))
            wipe = future & (np.arange(len(changed)) % 3 == 0)
            changed.loc[wipe, core.RAW] = np.nan
            alt = core.features(changed).set_index('row_id')[core.FULL]
            assert same(te_features.loc[b.loc[past, 'row_id']].to_numpy(),
                        alt.loc[b.loc[past, 'row_id']].to_numpy())
            assert same(tr_features.to_numpy(), core.features(train).set_index('row_id')[core.FULL].to_numpy())
            records.append(dict(kind='future', farm=farm, cut=cut, past_rows=int(past.sum()),
                                changed_future_rows=int(future.sum())))
        changed = test.copy()
        other = ~f
        changed.loc[other, core.RAW] = changed.loc[other, core.RAW] * 7 + 200
        alt = core.features(changed).set_index('row_id')[core.FULL]
        assert same(te_features.loc[b.loc[f, 'row_id']].to_numpy(),
                    alt.loc[b.loc[f, 'row_id']].to_numpy())
        records.append(dict(kind='other_farm', farm=farm, preserved_rows=int(f.sum())))

    result = dict(status='PASS', input_hashes={name: core.sha(env.DATA / name)
                                               for name in ('train_X.csv', 'train_y.csv', 'test_X.csv')},
                  code_hash=core.sha(HERE), core_hash=core.sha(V1 / 'run.py'),
                  protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
                  train_rows=len(train), test_rows=len(test), test_days=int(b.groupby(['farm', 'day']).ngroups),
                  feature_count=len(core.FULL), test_row_order='PASS',
                  train_test_combined_parity='PASS', checks=records)
    (out / 'features_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
