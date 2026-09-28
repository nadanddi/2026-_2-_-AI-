"""Evaluate the exact three-seed EC ensemble used by the local candidate."""
import importlib.util
import json
import platform
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
BAG = HERE.parents[1] / 'tabpfn_cpu_bag'
spec = importlib.util.spec_from_file_location('three_seed_ensemble_core', BAG / 'run.py')
bag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bag)
core, previous, env = bag.core, bag.previous, bag.env
np, pd = bag.np, bag.pd
OLD = ROOT / 'local/tabpfn_cpu_bag/20260927_182052'
NEW = ROOT / 'local/tabpfn_cpu_bag_extension/20260928_013953'
THIRD = ROOT / 'local/tabpfn_cpu_bag_baseline2024/20260928_032031'


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def resample(frame, unit, seed):
    rows = frame.copy()
    rows['base_sq'] = (rows.sub_ec - rows.baseline) ** 2
    rows['blend_sq'] = (rows.sub_ec - rows.blend) ** 2
    groups = {farm: group.groupby(unit)[['base_sq', 'blend_sq']].sum().to_numpy(float)
              for farm, group in rows.groupby('farm')}
    rng = np.random.default_rng(seed)
    changes = np.empty(5000)
    for i in range(len(changes)):
        sampled = sum(table[rng.integers(0, len(table), len(table))].sum(axis=0)
                      for table in groups.values())
        changes[i] = np.sqrt(sampled[1]/sampled[0]) - 1
    return dict(block_counts={f: len(t) for f, t in groups.items()},
                draws=len(changes), seed=seed,
                ci95=[float(x) for x in np.quantile(changes, [.025,.975])],
                ci99=[float(x) for x in np.quantile(changes, [.005,.995])],
                share_improved=float(np.mean(changes<0)))


def main():
    out = ROOT / 'local/ec_three_seed_ensemble_cv' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(code_hash=core.sha(HERE), protocol_hash=core.sha(HERE.with_name('PROTOCOL.md')),
                    core_hash=core.sha(HERE.parents[1]/'rl_ec_v1/run.py'),
                    cache_code_hash=core.sha(HERE.parents[1]/'rl_ec_v2/run.py'),
                    splits_hash=core.sha(bag.SPLITS),
                    bag_files={str(f): core.sha((OLD if f in (0,2) else NEW)/f'fold{f}_bag.csv')
                               for f in (0,2,4,6)},
                    third_files={str(f): core.sha(THIRD/f'fold{f}.csv') for f in (0,2,4,6)},
                    inputs={n:core.sha(env.DATA/n) for n in ('train_X.csv','train_y.csv')},
                    python=platform.python_version(), numpy=np.__version__,
                    folds=(0,2,4,6), seeds=(7,101,2024), context_seeds=(1,2,3,4),
                    blend=(.8,.2), shrink=.5)
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    dev = bag.prepare()
    frames=[]
    scores={}
    for fold in (0,2,4,6):
        trm,vam=core.split(dev,fold)
        tr,va=dev[trm],dev[vam].reset_index(drop=True)
        source=(OLD if fold in (0,2) else NEW)/f'fold{fold}_bag.csv'
        saved=pd.read_csv(source)
        assert saved.row_id.tolist()==va.row_id.tolist()
        third=pd.read_csv(THIRD/f'fold{fold}.csv')
        assert third.row_id.tolist()==va.row_id.tolist()
        members=[]
        for seed in (7,101,2024):
            cache=ROOT/'local/ec_baseline_cache'/(previous.cache_key(tr,va,seed)+'.npz')
            assert cache.is_file(), f'Missing exact-input cache {fold}/{seed}'
            with np.load(cache) as z:
                assert np.array_equal(z['row_id'],va.row_id.to_numpy(str))
                mem=[z[n].copy() for n in ('et','lgb','mlp')]
            observed=(third.baseline if seed==2024 else saved[f'baseline_{seed}']).to_numpy(float)
            np.testing.assert_allclose(core.finish(mem,tr,va),observed,rtol=0,atol=1e-12)
            members.append(mem)
        raw=(.6*np.mean([m[0] for m in members],axis=0)
             +.3*np.mean([m[1] for m in members],axis=0)
             +.1*np.mean([m[2] for m in members],axis=0))
        base=np.clip(core.shrink(raw,va),tr.sub_ec.min(),tr.sub_ec.max())
        blend=np.clip(core.shrink(.8*raw+.2*saved.bag.to_numpy(float),va),
                      tr.sub_ec.min(),tr.sub_ec.max())
        a,b=rmse(va.sub_ec,base),rmse(va.sub_ec,blend)
        scores[str(fold)]=dict(baseline=a,candidate=b,relative_change=b/a-1)
        row=va[['row_id','farm','day','hour','block','sub_ec']].copy()
        row['fold']=fold
        row['baseline']=base
        row['blend']=blend
        frames.append(row)
        row.to_csv(out/f'fold{fold}.csv',index=False,float_format='%.17g')
        print(f'FOLD {fold}: {a:.6f} -> {b:.6f} ({b/a-1:+.3%})',flush=True)
    combined=pd.concat(frames,ignore_index=True)
    assert combined.row_id.is_unique
    base,blend=rmse(combined.sub_ec,combined.baseline),rmse(combined.sub_ec,combined.blend)
    by_farm={f:dict(baseline=rmse(g.sub_ec,g.baseline),candidate=rmse(g.sub_ec,g.blend))
             for f,g in combined.groupby('farm')}
    for row in by_farm.values():
        row['relative_change']=row['candidate']/row['baseline']-1
    day=combined.groupby(['farm','day']).apply(
        lambda g: pd.Series({'base':rmse(g.sub_ec,g.baseline),'blend':rmse(g.sub_ec,g.blend),
                             'gain_sq':float(((g.sub_ec-g.baseline)**2-(g.sub_ec-g.blend)**2).sum())}),
        include_groups=False).reset_index()
    positive=day.gain_sq[day.gain_sq>0].sort_values(ascending=False)
    day_stats=dict(total_days=len(day),improved_days=int((day['blend']<day['base']).sum()),
                   median_relative_change=float(np.median(day['blend']/day['base']-1)),
                   top5_share_of_positive_gain=float(positive.head(5).sum()/positive.sum()),
                   top10_share_of_positive_gain=float(positive.head(10).sum()/positive.sum()))
    result=dict(scores=scores,pooled=dict(baseline=base,candidate=blend,relative_change=blend/base-1),
                by_farm=by_farm,day_stats=day_stats,
                bootstrap_day=resample(combined,'day',280926),
                bootstrap_five_day=resample(combined,'block',280927),
                passes_search=bool(all(s['relative_change']<0 for s in scores.values()) and blend/base-1<=-.01),
                confirmation_scored=False,hidden_labels_read=False,adopted=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with bag.threadpool_limits(limits=4):
        main()
