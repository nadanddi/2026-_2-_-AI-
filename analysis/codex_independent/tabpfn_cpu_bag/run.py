"""Offline CPU reproduction of four-context TabPFN EC blend."""
import sys
import importlib.util
import json
import platform
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
PROBE = HERE.parents[1] / 'tabpfn_cpu_probe'
spec = importlib.util.spec_from_file_location('tabpfn_cpu_probe_runner', PROBE / 'run.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
core, previous, env = probe.core, probe.previous, probe.env
TabPFNRegressor, ModelVersion, torch = probe.TabPFNRegressor, probe.ModelVersion, probe.torch
np, pd, tabpfn = probe.np, probe.pd, probe.tabpfn
threadpool_limits = probe.threadpool_limits

ROOT = HERE.parents[2]
SPLITS = ROOT / 'local/rl_ec_v1/20260927_173801/splits.csv'
PROBE_OUTPUT = ROOT / 'local/tabpfn_cpu_probe/20260927_181604'
FOLDS, CONTEXT_SEEDS, BASE_SEEDS = (0, 2), (1, 2, 3, 4), (7, 101)


def prepare():
    raw = pd.read_csv(env.DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(['F13', 'F47'])].reset_index(drop=True)
    y = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    lab = core.features(raw).merge(y, on='row_id', validate='one_to_one').dropna(subset=['sub_ec'])
    lab = lab.merge(pd.read_csv(SPLITS)[['row_id', 'fold', 'block']], on='row_id', validate='one_to_one')
    allowed = core.split(lab, 8)[0] & core.split(lab, 9)[0]
    return lab[allowed].reset_index(drop=True)


def main():
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).resolve()
        assert out.is_dir() and ROOT / 'local/tabpfn_cpu_bag' in out.parents
    else:
        out = ROOT / 'local/tabpfn_cpu_bag' / datetime.now().strftime('%Y%m%d_%H%M%S')
        out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    def log(message):
        print(message, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')
    manifest = dict(code_hash=core.sha(__file__), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
        probe_code_hash=core.sha(PROBE / 'run.py'),
        v1_hash=core.sha(HERE.parents[1]/'rl_ec_v1/run.py'),
        v2_hash=core.sha(HERE.parents[1]/'rl_ec_v2/run.py'),
        splits_hash=core.sha(SPLITS),
        inputs={n: core.sha(env.DATA/n) for n in ['train_X.csv', 'train_y.csv']},
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        tabpfn=tabpfn.__version__, offline=True, device='cpu', precision='float32',
        folds=FOLDS, context_seeds=CONTEXT_SEEDS, base_seeds=BASE_SEEDS,
        context_size=2000, n_estimators=4)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding='utf-8')) == manifest, 'Resume requires exact setup'
        log('RESUME ' + str(out))
    else:
        save('manifest.json', manifest)
        log('OUTPUT ' + str(out))
    dev = prepare()
    log(f'SEARCH ROWS {len(dev)}; locked confirmation excluded')
    start = time.perf_counter()
    results = {}
    for fold in FOLDS:
        trm, vam = core.split(dev, fold)
        tr, va = dev[trm], dev[vam].reset_index(drop=True)
        Xtr = tr[core.FULL].to_numpy(dtype=np.float32)
        Xva = va[core.FULL].to_numpy(dtype=np.float32)
        ytr = tr.sub_ec.to_numpy(float)
        members = []
        for seed in CONTEXT_SEEDS:
            file = out / f'fold{fold}_context{seed}.csv'
            if file.exists():
                cached = pd.read_csv(file)
                assert cached.row_id.tolist() == va.row_id.tolist(), 'Context checkpoint row mismatch'
                member = cached.member.to_numpy(float)
                log(f'REUSED fold={fold} context={seed}')
            elif fold == 0 and seed == 1:
                prior_manifest = json.loads((PROBE_OUTPUT/'manifest.json').read_text(encoding='utf-8'))
                for key in ['inputs', 'v1_hash', 'v2_hash', 'splits_hash', 'python', 'numpy', 'torch', 'tabpfn', 'offline']:
                    assert prior_manifest[key] == manifest[key], f'Probe mismatch: {key}'
                assert prior_manifest['context_seed'] == seed and prior_manifest['base_seed'] == 7
                assert prior_manifest['n_estimators'] == 4 and prior_manifest['context_size'] == 2000
                cached = pd.read_csv(PROBE_OUTPUT/'search_fold0_seed7_context1.csv')
                assert cached.row_id.tolist() == va.row_id.tolist(), 'Probe row mismatch'
                member = cached.member.to_numpy(float)
                log('REUSED exact probe fold=0 context=1')
                pd.DataFrame({'row_id': va.row_id, 'member': member}).to_csv(file, index=False, float_format='%.17g')
            else:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(tr), size=2000, replace=False)
                m = TabPFNRegressor.create_default_for_version(ModelVersion.V2,
                    device='cpu', n_estimators=4, random_state=seed,
                    ignore_pretraining_limits=True, inference_precision=torch.float32)
                m.fit(Xtr[idx], ytr[idx])
                member = np.asarray(m.predict(Xva), dtype=float)
                pd.DataFrame({'row_id': va.row_id, 'member': member}).to_csv(file, index=False, float_format='%.17g')
                log(f'PREDICT fold={fold} context={seed} done in {time.perf_counter()-start:.1f}s')
            assert np.isfinite(member).all()
            members.append(member)
        bag = np.mean(members, axis=0)
        row = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        row['bag'] = bag
        for base_seed in BASE_SEEDS:
            base = previous.get_baseline(tr, va, base_seed, log)
            base_pred = core.finish(base, tr, va)
            raw_base = .6*base[0] + .3*base[1] + .1*base[2]
            blend = np.clip(core.shrink(.8*raw_base + .2*bag, va), tr.sub_ec.min(), tr.sub_ec.max())
            a, b = core.rmse(va.sub_ec, base_pred), core.rmse(va.sub_ec, blend)
            results[f'fold{fold}_base{base_seed}'] = dict(baseline=a, candidate=b, relative_change=b/a-1)
            row[f'baseline_{base_seed}'] = base_pred
            row[f'blend_{base_seed}'] = blend
            log(f'SCORE fold={fold} base_seed={base_seed}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
        row.to_csv(out/f'fold{fold}_bag.csv', index=False)
        save('partial_results.json', results)
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
    with threadpool_limits(limits=4):
        main()
