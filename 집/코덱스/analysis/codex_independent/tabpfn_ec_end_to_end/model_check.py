"""Checkpointed CPU inference checks for the fixed four-context EC member."""
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
PROBE = HERE.parents[1] / 'tabpfn_cpu_probe'
spec = importlib.util.spec_from_file_location('tabpfn_cpu_probe_e2e', PROBE / 'run.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
core, env = probe.core, probe.env
np, pd, torch, tabpfn = probe.np, probe.pd, probe.torch, probe.tabpfn
TabPFNRegressor, ModelVersion = probe.TabPFNRegressor, probe.ModelVersion
FEATURES_OUTPUT = ROOT / 'local/tabpfn_ec_end_to_end/20260928_021653/features_result.json'


def same(a, b):
    return bool(np.array_equal(a, b, equal_nan=True))


def main():
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).resolve()
        assert out.is_dir() and ROOT / 'local/tabpfn_ec_end_to_end' in out.parents
    else:
        out = ROOT / 'local/tabpfn_ec_end_to_end' / datetime.now().strftime('%Y%m%d_%H%M%S')
        out.mkdir(parents=True, exist_ok=False)

    def save(name, item):
        (out / name).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(msg + '\n')

    checked = json.loads(FEATURES_OUTPUT.read_text(encoding='utf-8'))
    assert checked['status'] == 'PASS'
    manifest = dict(code_hash=core.sha(HERE), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
                    feature_check_hash=core.sha(FEATURES_OUTPUT),
                    probe_code_hash=core.sha(PROBE / 'run.py'),
                    core_code_hash=core.sha(HERE.parents[1] / 'rl_ec_v1/run.py'),
                    input_hashes={n: core.sha(env.DATA / n) for n in ('train_X.csv', 'train_y.csv', 'test_X.csv')},
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    tabpfn=tabpfn.__version__, device='cpu', precision='float32',
                    offline=True, context_size=2000, n_estimators=4, seeds=(1, 2, 3, 4))
    assert manifest['input_hashes'] == checked['input_hashes']
    frozen = json.loads(json.dumps(manifest))
    if (out / 'manifest.json').exists():
        assert json.loads((out / 'manifest.json').read_text(encoding='utf-8')) == frozen
        log('RESUME ' + str(out))
    else:
        save('manifest.json', manifest)
        log('OUTPUT ' + str(out))

    train = pd.read_csv(env.DATA / 'train_X.csv')
    train = train[train.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    label = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    tr = core.features(train).merge(label, on='row_id', validate='one_to_one').dropna(subset=['sub_ec'])
    test = pd.read_csv(env.DATA / 'test_X.csv')
    te = core.features(test).set_index('row_id').loc[test.row_id].reset_index()
    row_ids = te.row_id.to_numpy(str)
    Xtr, Xte = tr[core.FULL].to_numpy(np.float32), te[core.FULL].to_numpy(np.float32)
    ytr = tr.sub_ec.to_numpy(float)
    ident = core.identify(test)
    time_id = ident.day * 24 + ident.hour
    farm = ident.farm.eq('F13')
    cut = int(time_id[farm].quantile(.5))
    past = (farm & time_id.le(cut)).to_numpy()
    future = (farm & time_id.gt(cut)).to_numpy()
    changed = test.copy()
    rng = np.random.default_rng(731)
    for col in core.RAW:
        vals = changed.loc[future, col].to_numpy(float)
        changed.loc[future, col] = vals * 13 + rng.normal(400, 10, len(vals))
    changed.loc[future & (np.arange(len(test)) % 3 == 0), core.RAW] = np.nan
    altered = core.features(changed).set_index('row_id').loc[test.row_id].reset_index()
    Xalt = altered[core.FULL].to_numpy(np.float32)
    assert same(Xte[past], Xalt[past])
    assert not same(Xte[future], Xalt[future])
    log(f'FEATURES train={len(tr)} test={len(te)}; F13 cut={cut}, past={past.sum()}, future={future.sum()}')

    def get(file, model, data, row_order):
        path = out / file
        if path.exists():
            with np.load(path) as z:
                assert np.array_equal(z['row_id'], row_order)
                prediction = z['prediction'].copy()
            log('REUSED ' + file)
        else:
            prediction = np.asarray(model.predict(data), dtype=float)
            assert prediction.shape == (len(row_order),) and np.isfinite(prediction).all()
            np.savez(path, row_id=row_order, prediction=prediction)
            log('PREDICT ' + file)
        return prediction

    start = time.perf_counter()
    records = {}
    originals = []
    futures = []
    for seed in (1, 2, 3, 4):
        files = [f'context{seed}_{kind}.npz' for kind in ('original', 'reverse', 'future')]
        if seed == 1:
            files.append('context1_repeat.npz')
        cached_all = all((out / file).exists() for file in files)
        if cached_all:
            model = None
        else:
            idx = np.random.default_rng(seed).choice(len(tr), size=2000, replace=False)
            model = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2, device='cpu', n_estimators=4, random_state=seed,
                ignore_pretraining_limits=True, inference_precision=torch.float32)
            model.fit(Xtr[idx], ytr[idx])
            log(f'FIT context={seed} elapsed={time.perf_counter()-start:.1f}s')
        original = get(files[0], model, Xte, row_ids)
        reversed_pred = get(files[1], model, Xte[::-1], row_ids[::-1])[::-1]
        future_pred = get(files[2], model, Xalt, row_ids)
        assert same(original, reversed_pred), f'Row order changed context={seed}'
        assert same(original[past], future_pred[past]), f'Future input changed past context={seed}'
        if seed == 1:
            # Independent second fit with the same selected context and model seed.
            if not (out / files[3]).exists():
                idx = np.random.default_rng(seed).choice(len(tr), size=2000, replace=False)
                repeat = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2, device='cpu', n_estimators=4, random_state=seed,
                    ignore_pretraining_limits=True, inference_precision=torch.float32)
                repeat.fit(Xtr[idx], ytr[idx])
                log(f'REFIT context=1 elapsed={time.perf_counter()-start:.1f}s')
            else:
                repeat = None
            repeated_pred = get(files[3], repeat, Xte, row_ids)
            assert same(original, repeated_pred), 'Independent refit differed'
        records[str(seed)] = dict(row_order='PASS', future_invariance='PASS',
                                  repeated_fit='PASS' if seed == 1 else 'not_run',
                                  min_prediction=float(original.min()),
                                  max_prediction=float(original.max()))
        originals.append(original)
        futures.append(future_pred)
        save('partial_results.json', dict(contexts=records, elapsed_seconds=time.perf_counter()-start))
        log(f'PASS context={seed} elapsed={time.perf_counter()-start:.1f}s')

    bag = np.mean(originals, axis=0)
    assert np.isfinite(bag).all()
    bag_future = np.mean(futures, axis=0)
    assert same(bag[past], bag_future[past])
    post = np.clip(core.shrink(bag, te), tr.sub_ec.min(), tr.sub_ec.max())
    post_future = np.clip(core.shrink(bag_future, altered), tr.sub_ec.min(), tr.sub_ec.max())
    assert same(post[past], post_future[past]), 'Causal shrink changed earlier predictions'
    np.savez(out / 'bag.npz', row_id=row_ids, prediction=bag)
    result = dict(status='PASS', context_checks=records, context_mean='PASS',
                  causal_shrink='PASS', feature_checks='PASS',
                  elapsed_seconds=time.perf_counter()-start,
                  test_labels_read=False, hidden_rmse_scored=False, adopted=False)
    save('result.json', result)
    log('FINISHED ' + json.dumps(result))


if __name__ == '__main__':
    torch.set_num_threads(4)
    with probe.threadpool_limits(limits=4):
        main()
