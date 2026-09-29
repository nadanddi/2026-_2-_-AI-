"""Offline CPU replication of the second preselected TabPFN context bag."""
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
BAG = HERE.parents[1] / 'tabpfn_cpu_bag'
spec = importlib.util.spec_from_file_location('cpu_bag_second_context', BAG / 'run.py')
bag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bag)
core, previous, env = bag.core, bag.previous, bag.env
np, pd, torch, tabpfn = bag.np, bag.pd, bag.torch, bag.tabpfn
TabPFNRegressor, ModelVersion = bag.TabPFNRegressor, bag.ModelVersion
FIRST_OLD = ROOT / 'local/tabpfn_cpu_bag/20260927_182052'
FIRST_NEW = ROOT / 'local/tabpfn_cpu_bag_extension/20260928_013953'
THIRD = ROOT / 'local/tabpfn_cpu_bag_baseline2024/20260928_032031'
FOLDS, CONTEXT_SEEDS, BASE_SEEDS = (0, 2, 4, 6), (5, 6, 7, 8), (7, 101, 2024)


def main():
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).resolve()
        assert out.is_dir() and ROOT / 'local/tabpfn_cpu_second_bag' in out.parents
    else:
        out = ROOT / 'local/tabpfn_cpu_second_bag' / datetime.now().strftime('%Y%m%d_%H%M%S')
        out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def log(message):
        print(message, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')

    manifest = dict(code_hash=core.sha(HERE), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
        first_old_hash=core.sha(FIRST_OLD / 'result.json'),
        first_new_hash=core.sha(FIRST_NEW / 'result.json'),
        third_hash=core.sha(THIRD / 'result.json'),
        bag_code_hash=core.sha(BAG / 'run.py'),
        v1_hash=core.sha(HERE.parents[1] / 'rl_ec_v1/run.py'),
        v2_hash=core.sha(HERE.parents[1] / 'rl_ec_v2/run.py'),
        splits_hash=core.sha(bag.SPLITS),
        inputs={n: core.sha(env.DATA / n) for n in ('train_X.csv', 'train_y.csv')},
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        tabpfn=tabpfn.__version__, device='cpu', precision='float32', offline=True,
        folds=FOLDS, context_seeds=CONTEXT_SEEDS, base_seeds=BASE_SEEDS,
        context_size=2000, n_estimators=4)
    frozen = json.loads(json.dumps(manifest))
    if (out / 'manifest.json').exists():
        assert json.loads((out / 'manifest.json').read_text(encoding='utf-8')) == frozen
        log('RESUME ' + str(out))
    else:
        save('manifest.json', manifest)
        log('OUTPUT ' + str(out))

    old = json.loads((FIRST_OLD / 'manifest.json').read_text(encoding='utf-8'))
    for key in ('inputs', 'v1_hash', 'v2_hash', 'splits_hash', 'python', 'numpy',
                'torch', 'tabpfn', 'device', 'precision', 'offline', 'context_size', 'n_estimators'):
        assert old[key] == frozen[key], f'First bag setup differs: {key}'
    dev = bag.prepare()
    log(f'SEARCH ROWS {len(dev)}; locked confirmation excluded')
    start = time.perf_counter()
    scores = {}
    for fold in FOLDS:
        trm, vam = core.split(dev, fold)
        tr, va = dev[trm], dev[vam].reset_index(drop=True)
        Xtr, Xva = tr[core.FULL].to_numpy(np.float32), va[core.FULL].to_numpy(np.float32)
        ytr = tr.sub_ec.to_numpy(float)
        members = []
        for seed in CONTEXT_SEEDS:
            file = out / f'fold{fold}_context{seed}.csv'
            if file.exists():
                saved = pd.read_csv(file)
                assert saved.row_id.tolist() == va.row_id.tolist()
                member = saved.member.to_numpy(float)
                log(f'REUSED fold={fold} context={seed}')
            else:
                idx = np.random.default_rng(seed).choice(len(tr), size=2000, replace=False)
                model = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2, device='cpu', n_estimators=4, random_state=seed,
                    ignore_pretraining_limits=True, inference_precision=torch.float32)
                model.fit(Xtr[idx], ytr[idx])
                member = np.asarray(model.predict(Xva), dtype=float)
                pd.DataFrame({'row_id': va.row_id, 'member': member}).to_csv(
                    file, index=False, float_format='%.17g')
                log(f'PREDICT fold={fold} context={seed} elapsed={time.perf_counter()-start:.1f}s')
            assert np.isfinite(member).all()
            members.append(member)
        second = np.mean(members, axis=0)
        first_file = (FIRST_OLD if fold in (0, 2) else FIRST_NEW) / f'fold{fold}_bag.csv'
        first = pd.read_csv(first_file)
        assert first.row_id.tolist() == va.row_id.tolist()
        row = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        row['second_bag'] = second
        row['first_bag'] = first.bag.to_numpy(float)
        for base_seed in BASE_SEEDS:
            base = previous.get_baseline(tr, va, base_seed, log)
            baseline = core.finish(base, tr, va)
            raw = .6 * base[0] + .3 * base[1] + .1 * base[2]
            candidate = np.clip(core.shrink(.8 * raw + .2 * second, va),
                                tr.sub_ec.min(), tr.sub_ec.max())
            first_candidate = np.clip(core.shrink(.8 * raw + .2 * first.bag.to_numpy(float), va),
                                      tr.sub_ec.min(), tr.sub_ec.max())
            a, b, c = (core.rmse(va.sub_ec, p) for p in (baseline, candidate, first_candidate))
            scores[f'fold{fold}_base{base_seed}'] = dict(
                baseline=a, second_bag=b, first_bag=c,
                relative_change=b/a-1, second_vs_first=b/c-1)
            row[f'baseline_{base_seed}'] = baseline
            row[f'second_{base_seed}'] = candidate
            row[f'first_{base_seed}'] = first_candidate
            log(f'SCORE fold={fold} base={base_seed}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%}); vs first {b/c-1:+.3%}')
        row.to_csv(out / f'fold{fold}_bag.csv', index=False, float_format='%.17g')
        save('partial_results.json', scores)
    changes = [r['relative_change'] for r in scores.values()]
    result = dict(scores=scores, all_improve=bool(all(v < 0 for v in changes)),
                  mean_relative_change=float(np.mean(changes)),
                  passes_search=bool(all(v < 0 for v in changes) and np.mean(changes) <= -.01),
                  elapsed_seconds=time.perf_counter()-start,
                  confirmation_scored=False, adopted=False)
    save('result.json', result)
    log('FINISHED ' + json.dumps(result))


if __name__ == '__main__':
    torch.set_num_threads(4)
    with bag.threadpool_limits(limits=4):
        main()
