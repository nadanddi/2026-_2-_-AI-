"""Fixed-setting CPU TabPFN replication on additional EC search folds."""
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
BAG = HERE.parents[1] / 'tabpfn_cpu_bag'
spec = importlib.util.spec_from_file_location('previous_bag', BAG / 'run.py')
bag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bag)
core, previous, env = bag.core, bag.previous, bag.env
np, pd, torch, tabpfn = bag.np, bag.pd, bag.torch, bag.tabpfn
TabPFNRegressor, ModelVersion = bag.TabPFNRegressor, bag.ModelVersion
ROOT = HERE.parents[2]
OLD_OUTPUT = ROOT / 'local/tabpfn_cpu_bag/20260927_182052'
FOLDS, CONTEXT_SEEDS, BASE_SEEDS = (4, 6), (1, 2, 3, 4), (7, 101)


def main():
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).resolve()
        assert out.is_dir() and ROOT / 'local/tabpfn_cpu_bag_extension' in out.parents
    else:
        out = ROOT / 'local/tabpfn_cpu_bag_extension' / datetime.now().strftime('%Y%m%d_%H%M%S')
        out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def log(message):
        print(message, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')

    manifest = dict(
        code_hash=core.sha(HERE), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
        previous_bag_code_hash=core.sha(BAG / 'run.py'),
        previous_bag_manifest_hash=core.sha(OLD_OUTPUT / 'manifest.json'),
        previous_bag_result_hash=core.sha(OLD_OUTPUT / 'result.json'),
        previous_bag_predictions={f'fold{f}': core.sha(OLD_OUTPUT / f'fold{f}_bag.csv') for f in (0, 2)},
        v1_hash=core.sha(HERE.parents[1] / 'rl_ec_v1/run.py'),
        v2_hash=core.sha(HERE.parents[1] / 'rl_ec_v2/run.py'),
        splits_hash=core.sha(bag.SPLITS),
        inputs={n: core.sha(env.DATA / n) for n in ('train_X.csv', 'train_y.csv')},
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        tabpfn=tabpfn.__version__, offline=True, device='cpu', precision='float32',
        folds=FOLDS, context_seeds=CONTEXT_SEEDS, base_seeds=BASE_SEEDS,
        context_size=2000, n_estimators=4,
    )
    frozen = json.loads(json.dumps(manifest))
    if (out / 'manifest.json').exists():
        assert json.loads((out / 'manifest.json').read_text(encoding='utf-8')) == frozen
        log('RESUME ' + str(out))
    else:
        save('manifest.json', manifest)
        log('OUTPUT ' + str(out))

    old = json.loads((OLD_OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
    for key in ('inputs', 'v1_hash', 'v2_hash', 'splits_hash', 'python', 'numpy',
                'torch', 'tabpfn', 'offline', 'device', 'precision', 'context_seeds',
                'base_seeds', 'context_size', 'n_estimators'):
        assert old[key] == frozen[key], f'Prior setup differs: {key}'
    assert old['folds'] == [0, 2]
    dev = bag.prepare()
    log(f'SEARCH ROWS {len(dev)}; locked confirmation excluded')
    results = {}
    start = time.perf_counter()
    for fold in FOLDS:
        trm, vam = core.split(dev, fold)
        tr, va = dev[trm], dev[vam].reset_index(drop=True)
        Xtr, Xva = tr[core.FULL].to_numpy(dtype=np.float32), va[core.FULL].to_numpy(dtype=np.float32)
        ytr = tr.sub_ec.to_numpy(float)
        members = []
        for seed in CONTEXT_SEEDS:
            file = out / f'fold{fold}_context{seed}.csv'
            if file.exists():
                cached = pd.read_csv(file)
                assert cached.row_id.tolist() == va.row_id.tolist()
                member = cached.member.to_numpy(float)
                log(f'REUSED fold={fold} context={seed}')
            else:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(tr), size=2000, replace=False)
                model = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2, device='cpu', n_estimators=4, random_state=seed,
                    ignore_pretraining_limits=True, inference_precision=torch.float32)
                model.fit(Xtr[idx], ytr[idx])
                member = np.asarray(model.predict(Xva), dtype=float)
                pd.DataFrame({'row_id': va.row_id, 'member': member}).to_csv(
                    file, index=False, float_format='%.17g')
                log(f'PREDICT fold={fold} context={seed} done in {time.perf_counter()-start:.1f}s')
            assert np.isfinite(member).all()
            members.append(member)
        bag_pred = np.mean(members, axis=0)
        row = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        row['bag'] = bag_pred
        for base_seed in BASE_SEEDS:
            base = previous.get_baseline(tr, va, base_seed, log)
            base_pred = core.finish(base, tr, va)
            raw_base = .6 * base[0] + .3 * base[1] + .1 * base[2]
            blend = np.clip(core.shrink(.8 * raw_base + .2 * bag_pred, va),
                            tr.sub_ec.min(), tr.sub_ec.max())
            a, b = core.rmse(va.sub_ec, base_pred), core.rmse(va.sub_ec, blend)
            results[f'fold{fold}_base{base_seed}'] = dict(
                baseline=a, candidate=b, relative_change=b/a-1)
            row[f'baseline_{base_seed}'] = base_pred
            row[f'blend_{base_seed}'] = blend
            log(f'SCORE fold={fold} base_seed={base_seed}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
        row.to_csv(out / f'fold{fold}_bag.csv', index=False, float_format='%.17g')
        save('partial_results.json', results)

    old_results = json.loads((OLD_OUTPUT / 'result.json').read_text(encoding='utf-8'))['scores']
    for fold in (0, 2):
        trm, vam = core.split(dev, fold)
        va = dev[vam].reset_index(drop=True)
        row = pd.read_csv(OLD_OUTPUT / f'fold{fold}_bag.csv')
        assert row.row_id.tolist() == va.row_id.tolist()
        for seed in BASE_SEEDS:
            key = f'fold{fold}_base{seed}'
            a = core.rmse(row.sub_ec, row[f'baseline_{seed}'])
            b = core.rmse(row.sub_ec, row[f'blend_{seed}'])
            assert abs(a-old_results[key]['baseline']) < 1e-12
            assert abs(b-old_results[key]['candidate']) < 1e-12
            results[key] = dict(baseline=a, candidate=b, relative_change=b/a-1)
    diffs = [v['relative_change'] for v in results.values()]
    result = dict(scores=results, all_improve=all(v < 0 for v in diffs),
                  mean_relative_change=float(np.mean(diffs)),
                  passes_search=all(v < 0 for v in diffs) and np.mean(diffs) <= -.01,
                  elapsed_seconds=time.perf_counter()-start,
                  confirmation_scored=False, adopted=False)
    save('result.json', result)
    log('FINISHED ' + json.dumps(result))


if __name__ == '__main__':
    torch.set_num_threads(4)
    with bag.threadpool_limits(limits=4):
        main()
