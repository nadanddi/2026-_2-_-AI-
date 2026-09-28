"""Build an auditable local candidate without scoring or submitting it."""
import importlib.util
import json
import platform
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
V1 = HERE.parents[1] / 'rl_ec_v1'
import sys
sys.path.insert(0, str(V1))
import env  # noqa: E402
spec = importlib.util.spec_from_file_location('ec_core_local_candidate', V1 / 'run.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
np, pd = core.np, core.pd

PARITY = ROOT / 'local/ec_submission_parity/20260928_031746'
E2E = ROOT / 'local/tabpfn_ec_end_to_end/20260928_022031'
OLD = env.SOURCE / 'research/submissions'
WEIGHTS = Path('C:/Users/aozks/AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt')


def main():
    out = ROOT / 'local/ec_tabpfn_local_candidate' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((PARITY / 'result.json').read_text(encoding='utf-8'))
    checked = json.loads((E2E / 'result.json').read_text(encoding='utf-8'))
    assert prior['status'] == checked['status'] == 'PASS'
    assert prior['exact_six_decimals'] and checked['causal_shrink'] == 'PASS'
    assert core.sha(WEIGHTS) == '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'

    s04 = pd.read_csv(OLD / 'submission_04.csv')
    s06 = pd.read_csv(OLD / 'submission_06.csv')
    s07 = pd.read_csv(OLD / 'submission_07.csv')
    test = pd.read_csv(env.DATA / 'test_X.csv')
    sample = pd.read_csv(env.DATA / 'sample_submission.csv')
    ids = test.row_id.to_numpy(str)
    assert len(ids) == 1440 and len(set(ids)) == len(ids)
    for frame in (s04, s06, s07, sample):
        assert frame.row_id.tolist() == ids.tolist()
    assert np.array_equal(s06.sub_temp.to_numpy(float), s07.sub_temp.to_numpy(float))
    assert np.array_equal(s04.sub_ec.to_numpy(float), s07.sub_ec.to_numpy(float))

    with np.load(PARITY / 'ec_raw_and_final.npz') as z:
        assert np.array_equal(z['row_id'], ids)
        raw_base = z['raw'].copy()
        final_base = z['final'].copy()
    assert np.array_equal(np.round(final_base, 6), s04.sub_ec.to_numpy(float))
    with np.load(E2E / 'bag.npz') as z:
        assert np.array_equal(z['row_id'], ids)
        tabpfn_mean = z['prediction'].copy()
    assert np.isfinite(tabpfn_mean).all()

    te = core.features(test).set_index('row_id').loc[ids].reset_index()
    y = pd.read_csv(env.DATA / 'train_y.csv', usecols=['sub_ec']).sub_ec
    low, high = float(y.min()), float(y.max())
    mixed_raw = .8 * raw_base + .2 * tabpfn_mean
    mixed_shrunk = core.shrink(mixed_raw, te)
    pred_ec = np.clip(mixed_shrunk, low, high)
    candidate = pd.DataFrame({'row_id': ids, 'sub_temp': s06.sub_temp.to_numpy(float),
                              'sub_ec': pred_ec})
    assert candidate.columns.tolist() == ['row_id', 'sub_temp', 'sub_ec']
    assert np.isfinite(candidate[['sub_temp', 'sub_ec']].to_numpy(float)).all()
    assert (candidate.sub_ec > 0).all()
    candidate_path = out / 'candidate_temp06_ec_tabpfn_v2_cpu.csv'
    candidate.to_csv(candidate_path, index=False, float_format='%.6f', lineterminator='\n')
    reread = pd.read_csv(candidate_path)
    assert reread.columns.tolist() == ['row_id', 'sub_temp', 'sub_ec']
    assert reread.row_id.tolist() == sample.row_id.tolist()
    assert reread.row_id.is_unique and len(reread) == 1440
    assert np.isfinite(reread[['sub_temp', 'sub_ec']].to_numpy(float)).all().all()

    manifest = dict(code_hash=core.sha(HERE), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
                    core_hash=core.sha(V1 / 'run.py'),
                    inputs={n: core.sha(env.DATA / n) for n in
                            ('train_X.csv', 'train_y.csv', 'test_X.csv', 'sample_submission.csv')},
                    source_submissions={n: core.sha(OLD / n) for n in
                                        ('submission_04.csv', 'submission_06.csv', 'submission_07.csv')},
                    baseline_raw=core.sha(PARITY / 'ec_raw_and_final.npz'),
                    tabpfn_mean=core.sha(E2E / 'bag.npz'), tabpfn_weight_hash=core.sha(WEIGHTS),
                    candidate_hash=core.sha(candidate_path), python=platform.python_version(),
                    numpy=np.__version__, ec_blend=dict(baseline=.8, tabpfn=.2),
                    shrink=.5, clip=[low, high], hidden_labels_read=False,
                    platform_submitted=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    d = pred_ec - final_base
    by_farm = {}
    farms = core.identify(test).farm
    for farm in ('F13', 'F47'):
        mask = farms.eq(farm).to_numpy()
        by_farm[farm] = dict(count=int(mask.sum()), mean_delta=float(d[mask].mean()),
                             rms_delta=float(np.sqrt(np.mean(d[mask] ** 2))),
                             min_candidate=float(pred_ec[mask].min()),
                             max_candidate=float(pred_ec[mask].max()))
    result = dict(status='DRAFT', rows=len(candidate), file=str(candidate_path),
                  sha256=core.sha(candidate_path), reference_ec_parity='PASS',
                  temperature_unchanged='PASS', format='PASS',
                  ec_mean=float(pred_ec.mean()), ec_min=float(pred_ec.min()),
                  ec_max=float(pred_ec.max()), clip_count=int(np.sum(mixed_shrunk != pred_ec)),
                  mean_delta_vs_submission04=float(d.mean()),
                  rms_delta_vs_submission04=float(np.sqrt(np.mean(d ** 2))),
                  by_farm=by_farm, hidden_rmse_scored=False,
                  platform_submitted=False, adopted=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
