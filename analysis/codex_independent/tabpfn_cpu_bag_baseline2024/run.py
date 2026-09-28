"""Compare fixed TabPFN bag against the third original EC baseline seed."""
import importlib.util
import json
import platform
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
BAG = HERE.parents[1] / 'tabpfn_cpu_bag'
spec = importlib.util.spec_from_file_location('cpu_bag_base2024', BAG / 'run.py')
bag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bag)
core, previous = bag.core, bag.previous
np, pd = bag.np, bag.pd
OLD = ROOT / 'local/tabpfn_cpu_bag/20260927_182052'
NEW = ROOT / 'local/tabpfn_cpu_bag_extension/20260928_013953'


def main():
    out = ROOT / 'local/tabpfn_cpu_bag_baseline2024' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def log(message):
        print(message, flush=True)
        with (out / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(message + '\n')

    old_manifest = json.loads((OLD / 'manifest.json').read_text(encoding='utf-8'))
    new_manifest = json.loads((NEW / 'manifest.json').read_text(encoding='utf-8'))
    for key in ('inputs', 'v1_hash', 'v2_hash', 'splits_hash', 'python', 'numpy'):
        assert old_manifest[key] == new_manifest[key]
    save('manifest.json', dict(code_hash=core.sha(HERE),
        protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
        old_manifest_hash=core.sha(OLD / 'manifest.json'),
        new_manifest_hash=core.sha(NEW / 'manifest.json'),
        old_predictions={str(f): core.sha(OLD / f'fold{f}_bag.csv') for f in (0, 2)},
        new_predictions={str(f): core.sha(NEW / f'fold{f}_bag.csv') for f in (4, 6)},
        input_hashes=old_manifest['inputs'], splits_hash=old_manifest['splits_hash'],
        python=platform.python_version(), numpy=np.__version__,
        fold_ids=(0, 2, 4, 6), baseline_seed=2024, contexts=(1, 2, 3, 4),
        confirmation_scored=False))
    dev = bag.prepare()
    scores = {}
    farms = {}
    for fold in (0, 2, 4, 6):
        trm, vam = core.split(dev, fold)
        tr, va = dev[trm], dev[vam].reset_index(drop=True)
        saved = pd.read_csv((OLD if fold in (0, 2) else NEW) / f'fold{fold}_bag.csv')
        assert saved.row_id.tolist() == va.row_id.tolist()
        member = previous.get_baseline(tr, va, 2024, log)
        baseline = core.finish(member, tr, va)
        raw = .6 * member[0] + .3 * member[1] + .1 * member[2]
        blend = np.clip(core.shrink(.8 * raw + .2 * saved.bag.to_numpy(float), va),
                        tr.sub_ec.min(), tr.sub_ec.max())
        a, b = core.rmse(va.sub_ec, baseline), core.rmse(va.sub_ec, blend)
        scores[str(fold)] = dict(baseline=a, candidate=b, relative_change=b/a-1)
        farms[str(fold)] = {}
        for farm in ('F13', 'F47'):
            select = va.farm.eq(farm).to_numpy()
            fa, fb = core.rmse(va.sub_ec[select], baseline[select]), core.rmse(va.sub_ec[select], blend[select])
            farms[str(fold)][farm] = dict(baseline=fa, candidate=fb, relative_change=fb/fa-1)
        pd.DataFrame({'row_id': va.row_id, 'farm': va.farm, 'day': va.day,
                      'sub_ec': va.sub_ec, 'baseline': baseline, 'blend': blend}).to_csv(
                          out / f'fold{fold}.csv', index=False, float_format='%.17g')
        save('partial_results.json', dict(scores=scores, by_farm=farms))
        log(f'FOLD {fold}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})')
    result = dict(scores=scores, by_farm=farms, all_improve=bool(all(v['relative_change'] < 0 for v in scores.values())),
                  mean_relative_change=float(np.mean([v['relative_change'] for v in scores.values()])),
                  confirmation_scored=False, adopted=False)
    save('result.json', result)
    log('FINISHED')


if __name__ == '__main__':
    with bag.threadpool_limits(limits=4):
        main()
