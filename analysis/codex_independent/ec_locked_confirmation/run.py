"""One-use, offline confirmation of the frozen three-seed EC candidate."""
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
spec = importlib.util.spec_from_file_location('ec_confirmation_dependencies', BAG / 'run.py')
bag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bag)
core, previous, env = bag.core, bag.previous, bag.env
np, pd, torch, tabpfn = bag.np, bag.pd, bag.torch, bag.tabpfn
TabPFNRegressor, ModelVersion = bag.TabPFNRegressor, bag.ModelVersion
FOLDS = (8, 9)
CONTEXT_SEEDS = (1, 2, 3, 4)
BASE_SEEDS = (7, 101, 2024)


def prepare():
    raw = pd.read_csv(env.DATA / 'train_X.csv')
    raw = raw[raw.row_id.str[:3].isin(('F13', 'F47'))].reset_index(drop=True)
    y = pd.read_csv(env.DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    lab = core.features(raw).merge(y, on='row_id', validate='one_to_one')
    lab = lab.dropna(subset=['sub_ec'])
    lab = lab.merge(pd.read_csv(bag.SPLITS)[['row_id', 'fold', 'block']],
                    on='row_id', validate='one_to_one')
    train_mask = core.split(lab, 8)[0] & core.split(lab, 9)[0]
    locked = lab.fold.isin(FOLDS)
    assert not (train_mask & locked).any()
    tr = lab[train_mask].reset_index(drop=True)
    va = lab[locked].reset_index(drop=True)
    assert tr.row_id.tolist() == bag.prepare().row_id.tolist()
    assert va.row_id.is_unique and set(va.fold) == set(FOLDS)
    assert not set(tr.row_id).intersection(va.row_id)
    return tr, va


def main():
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).resolve()
        assert out.is_dir() and ROOT / 'local/ec_locked_confirmation' in out.parents
    else:
        out = ROOT / 'local/ec_locked_confirmation' / datetime.now().strftime('%Y%m%d_%H%M%S')
        out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def log(message):
        print(message, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')

    model_file = Path.home() / 'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'
    manifest = dict(code_hash=core.sha(HERE), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
                    bag_code_hash=core.sha(BAG / 'run.py'),
                    v1_hash=core.sha(HERE.parents[1] / 'rl_ec_v1/run.py'),
                    v2_hash=core.sha(HERE.parents[1] / 'rl_ec_v2/run.py'),
                    splits_hash=core.sha(bag.SPLITS),
                    inputs={n: core.sha(env.DATA / n) for n in ('train_X.csv', 'train_y.csv')},
                    checkpoint_hash=core.sha(model_file),
                    python=platform.python_version(), numpy=np.__version__,
                    torch=torch.__version__, tabpfn=tabpfn.__version__,
                    device='cpu', precision='float32', offline=True,
                    folds=FOLDS, context_seeds=CONTEXT_SEEDS,
                    base_seeds=BASE_SEEDS, context_size=2000, n_estimators=4)
    frozen = json.loads(json.dumps(manifest))
    if (out / 'manifest.json').exists():
        assert json.loads((out / 'manifest.json').read_text(encoding='utf-8')) == frozen
        log('RESUME ' + str(out))
    else:
        save('manifest.json', frozen)
        log('OUTPUT ' + str(out))

    tr, locked = prepare()
    log(f'CONFIRM TRAIN={len(tr)} LOCKED={len(locked)}')
    start = time.perf_counter()
    all_rows = []
    scores = {}
    for fold in FOLDS:
        va = locked[locked.fold.eq(fold)].reset_index(drop=True)
        Xtr, Xva = tr[core.FULL].to_numpy(np.float32), va[core.FULL].to_numpy(np.float32)
        ytr = tr.sub_ec.to_numpy(float)
        members = []
        for seed in CONTEXT_SEEDS:
            file = out / f'fold{fold}_context{seed}.csv'
            digest_file = out / f'fold{fold}_context{seed}.sha256'
            if file.exists():
                assert digest_file.exists() and digest_file.read_text(encoding='ascii').strip() == core.sha(file)
                saved = pd.read_csv(file)
                assert saved.row_id.tolist() == va.row_id.tolist()
                member = saved.member.to_numpy(float)
                log(f'REUSED fold={fold} context={seed}')
            else:
                idx = np.random.default_rng(seed).choice(len(tr), size=2000, replace=False)
                model = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2, device='cpu', n_estimators=4,
                    random_state=seed, ignore_pretraining_limits=True,
                    inference_precision=torch.float32)
                model.fit(Xtr[idx], ytr[idx])
                member = np.asarray(model.predict(Xva), dtype=float)
                pd.DataFrame({'row_id': va.row_id, 'member': member}).to_csv(
                    file, index=False, float_format='%.17g')
                digest_file.write_text(core.sha(file) + '\n', encoding='ascii')
                log(f'PREDICT fold={fold} context={seed} elapsed={time.perf_counter()-start:.1f}s')
            assert np.isfinite(member).all()
            members.append(member)
        bag_prediction = np.mean(members, axis=0)
        raw_members = []
        row = va[['row_id', 'farm', 'day', 'hour', 'block', 'sub_ec']].copy()
        for seed in BASE_SEEDS:
            baseline_members = previous.get_baseline(tr, va, seed, log)
            raw = .6 * baseline_members[0] + .3 * baseline_members[1] + .1 * baseline_members[2]
            raw_members.append(raw)
            baseline = np.clip(core.shrink(raw, va), tr.sub_ec.min(), tr.sub_ec.max())
            candidate = np.clip(core.shrink(.8 * raw + .2 * bag_prediction, va),
                                tr.sub_ec.min(), tr.sub_ec.max())
            a, b = core.rmse(va.sub_ec, baseline), core.rmse(va.sub_ec, candidate)
            row[f'baseline_{seed}'] = baseline
            row[f'candidate_{seed}'] = candidate
            scores[f'fold{fold}_base{seed}'] = dict(baseline=a, candidate=b,
                                                   relative_change=b/a-1)
            log(f'SCORE fold={fold} base={seed}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
        raw_mean = np.mean(raw_members, axis=0)
        row['baseline'] = np.clip(core.shrink(raw_mean, va), tr.sub_ec.min(), tr.sub_ec.max())
        row['candidate'] = np.clip(core.shrink(.8 * raw_mean + .2 * bag_prediction, va),
                                   tr.sub_ec.min(), tr.sub_ec.max())
        a, b = core.rmse(va.sub_ec, row.baseline), core.rmse(va.sub_ec, row.candidate)
        scores[f'fold{fold}_ensemble'] = dict(baseline=a, candidate=b, relative_change=b/a-1)
        log(f'ENSEMBLE fold={fold}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
        row.to_csv(out / f'fold{fold}.csv', index=False, float_format='%.17g')
        all_rows.append(row)
        save('partial_results.json', scores)

    combined = pd.concat(all_rows, ignore_index=True)
    a, b = core.rmse(combined.sub_ec, combined.baseline), core.rmse(combined.sub_ec, combined.candidate)
    result = dict(scores=scores, pooled=dict(baseline=a, candidate=b, relative_change=b/a-1),
                  passes_confirmation=bool(b/a-1 <= -.01 and
                                           all(scores[f'fold{f}_ensemble']['relative_change'] < 0
                                               for f in FOLDS)),
                  elapsed_seconds=time.perf_counter()-start,
                  test_x_scored=False, platform_submitted=False)
    save('result.json', result)
    log('FINISHED ' + json.dumps(result))


if __name__ == '__main__':
    torch.set_num_threads(4)
    with bag.threadpool_limits(limits=4):
        main()
