"""Recreate the round-three EC submission from the independent pipeline."""
import importlib.util
import json
import platform
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
V1 = HERE.parents[1] / 'rl_ec_v1'
spec = importlib.util.spec_from_file_location('ec_core_submission_parity', V1 / 'run.py')
# The source module requires env to be the first project import.
import sys
sys.path.insert(0, str(V1))
import env  # noqa: E402
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
np, pd = core.np, core.pd
ORIGINAL = env.SOURCE / 'research/submissions/submission_04.csv'


def main():
    out = ROOT / 'local/ec_submission_parity' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def log(message):
        print(message, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')

    save('manifest.json', dict(code_hash=core.sha(HERE), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
         core_hash=core.sha(V1 / 'run.py'), original_submission_hash=core.sha(ORIGINAL),
         input_hashes={n: core.sha(env.DATA / n) for n in ('train_X.csv', 'train_y.csv', 'test_X.csv')},
         python=platform.python_version(), numpy=np.__version__, seeds=(7, 101, 2024)))
    train = pd.read_csv(env.DATA / 'train_X.csv')
    train = train[train.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    label = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    tr = core.features(train).merge(label, on='row_id', validate='one_to_one').dropna(subset=['sub_ec'])
    test = pd.read_csv(env.DATA / 'test_X.csv')
    va = core.features(test).set_index('row_id').loc[test.row_id].reset_index()
    et_predictions, lgb_predictions, mlp_predictions = [], [], []
    for seed in (7, 101, 2024):
        members = core.members(tr, va, seed)
        et_predictions.append(members[0])
        lgb_predictions.append(members[1])
        mlp_predictions.append(members[2])
        log(f'SEED {seed} finished')
    raw = (.6 * np.mean(et_predictions, axis=0)
           + .3 * np.mean(lgb_predictions, axis=0)
           + .1 * np.mean(mlp_predictions, axis=0))
    pred = np.clip(core.shrink(raw, va), label.sub_ec.min(), label.sub_ec.max())
    saved = pd.read_csv(ORIGINAL)
    assert saved.row_id.tolist() == va.row_id.tolist()
    diff = pred - saved.sub_ec.to_numpy(float)
    exact_rounded = bool(np.array_equal(np.round(pred, 6), saved.sub_ec.to_numpy(float)))
    pd.DataFrame({'row_id': va.row_id, 'sub_ec': pred}).to_csv(
        out / 'ec_reproduced.csv', index=False, float_format='%.17g')
    np.savez(out / 'ec_raw_and_final.npz', row_id=va.row_id.to_numpy(str),
             raw=raw, final=pred)
    result = dict(status='PASS' if exact_rounded else 'FAIL', exact_six_decimals=exact_rounded,
                  rows=len(pred), max_absolute_difference=float(np.max(np.abs(diff))),
                  mean_absolute_difference=float(np.mean(np.abs(diff))),
                  hidden_labels_read=False)
    save('result.json', result)
    log('RESULT ' + json.dumps(result))
    assert exact_rounded, 'Independent EC baseline did not reproduce the submitted EC column'


if __name__ == '__main__':
    with core.threadpool_limits(limits=4):
        main()
