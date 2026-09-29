"""Pre-registered, public-OOF-only screen for fixed H1 calibration."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / 'analysis/local'
SEARCH = LOCAL / 'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = LOCAL / 'ec_locked_confirmation/20260928_044934'


def rmse(y, p):
    return float(np.sqrt(np.mean(np.square(np.asarray(y) - np.asarray(p)))))


def main():
    frames = []
    for fold in (0, 2, 4, 6, 8, 9):
        path = (SEARCH if fold < 8 else CONFIRM) / f'fold{fold}.csv'
        z = pd.read_csv(path)
        z['v2'] = z['blend'] if fold < 8 else z['candidate']
        z['fold'] = fold
        frames.append(z[['row_id', 'farm', 'day', 'fold', 'sub_ec', 'v2']])
    x = pd.concat(frames, ignore_index=True)
    assert len(x) == 5616 and x.row_id.is_unique
    assert x.groupby(['farm', 'day']).size().eq(24).all()
    assert np.isfinite(x[['sub_ec', 'v2']].to_numpy(float)).all()
    x['h1'] = x.v2 + .10 * np.maximum(x.v2 - .60, 0)

    metrics = {}
    for name, group in [('overall', x),
                        *[(f'fold{fold}', x[x.fold.eq(fold)]) for fold in (0, 2, 4, 6, 8, 9)],
                        *[(farm, x[x.farm.eq(farm)]) for farm in ('F13', 'F47')]]:
        a, b = rmse(group.sub_ec, group.v2), rmse(group.sub_ec, group.h1)
        metrics[name] = dict(rows=len(group), v2_rmse=a, h1_rmse=b,
                             relative_change=b/a-1, mse_change=b*b-a*a)

    # Build per-day MSE differences directly from paired row losses.
    x['delta_sq'] = np.square(x.sub_ec-x.h1)-np.square(x.sub_ec-x.v2)
    daily = x.groupby(['farm', 'day']).delta_sq.mean().to_numpy()
    assert len(daily) == 234
    rng = np.random.default_rng(2901)
    samples = np.empty(20000)
    for i in range(len(samples)):
        samples[i] = daily[rng.integers(0, len(daily), len(daily))].mean()
    ci = np.quantile(samples, [.025, .975]).tolist()
    passed = (all(metrics[f'fold{i}']['relative_change'] < 0 for i in (0,2,4,6,8,9))
              and all(metrics[f]['relative_change'] < 0 for f in ('F13','F47'))
              and ci[1] < 0)
    result = dict(formula='p + 0.10 * max(p - 0.60, 0)',
                  rows=len(x), days=len(daily), metrics=metrics,
                  day_bootstrap_mse_delta_95ci=ci,
                  day_bootstrap_p_worse=float(np.mean(samples >= 0)),
                  screen_pass=bool(passed),
                  fixed_hyperparameters=True, test_input_read=False,
                  new_lock_labels_read=False, submission_created=False)
    out = LOCAL / 'ec_high_level_calibration' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
