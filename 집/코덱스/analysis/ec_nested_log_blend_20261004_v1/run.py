"""Family19: learn one weight between already bounded LOG/v2 endpoints."""
from pathlib import Path
import sys, json, math, argparse, importlib.util, gc
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
sys.path.insert(0, str(ROOT / '집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

AN = ROOT / '집/코덱스/analysis'
LOCAL = ROOT / '집/코덱스/local'
OUT = LOCAL / H.name
LOG = LOCAL / 'ec_log_partition_mean_20261004_v1'
LOG_H = AN / 'ec_log_partition_mean_20261004_v1'
N_PATH = AN / 'ec_nested_high_specialist_20261004_v1/run_v2.py'
M_PATH = LOG_H / 'run.py'
N_SHA = '0ceb89e38692f76466000a4b3a83e056c99a04df10c519ddba59e70a55c2bbaf'
M_SHA = '26c098b2d9cc0b18b816f37e0d8eb0985ae4d828bbcb945d0341ecf453c421f8'
S_SHA = '82074e7a04ea681bc94e94e1c90354a743609e0f008b1a378a5a9a23575debed'
LAMBDA = .01
SEEDS = [7, 101, 2024]
assert S.sha(N_PATH) == N_SHA and S.sha(M_PATH) == M_SHA
assert S.sha(Path(S.__file__)) == S_SHA

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

N = module('nested_log_id_guards', N_PATH)
M = module('nested_log_ratio_recipe', M_PATH)

def finite(x):
    assert np.isfinite(np.asarray(x, float)).all()

def maxdiff(a, b, tol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.size > 0
    finite(a); finite(b)
    delta = float(np.max(np.abs(a - b)))
    assert delta < tol, delta
    return delta

def vector_id_sha(x):
    import hashlib
    return hashlib.sha256('\n'.join(str(z) for z in x).encode()).hexdigest()

def quant(x):
    x = np.asarray(x, float)
    finite(x)
    return dict(n=len(x), mean=float(np.mean(x)),
                quantiles=np.quantile(x, [0, .05, .5, .95, 1]).tolist())

def support(frame):
    out = []
    for (farm, late), g in frame.assign(late=frame.day >= 179).groupby(['farm', 'late']):
        days = g[['farm', 'day']].drop_duplicates()
        out.append(dict(farm=farm, late=bool(late), rows=len(g), days=len(days),
                        min_day=int(g.day.min()), max_day=int(g.day.max())))
    return out

def spans(tr):
    """Span in record day/season coordinates; no inferred calendar dates."""
    spans_out = []
    for farm, g in tr.groupby('farm'):
        daily = g.groupby('day').season.first().reset_index().sort_values('season')
        for i in range(len(daily)):
            window = daily.iloc[max(0, i - 10):min(len(daily), i + 11)]
            spans_out.append(dict(farm=farm, day=int(daily.day.iloc[i]),
                                  n_records=len(window),
                                  record_day_span=int(window.day.max() - window.day.min()),
                                  season_span=float(window.season.max() - window.season.min())))
    return spans_out

def endpoint(core, base, raw, old_raw, query, lo, hi):
    old_shrunk = core.shrink(old_raw, query)
    new_shrunk = core.shrink(raw, query)
    proposal = base + .48 * (new_shrunk - old_shrunk)
    return np.clip(proposal, lo, hi), old_shrunk, new_shrunk, proposal

def outer_reference(v, k, seed, tr, q, outer, core):
    path = LOG / f'{v}_{k}_{seed}.csv'
    d = pd.read_csv(path, float_precision='round_trip')
    assert len(d) == len(q) and d.row_id.is_unique
    assert np.array_equal(d.row_id, q.row_id)
    assert (d.validator == v).all() and (d.fold == k).all() and (d.seed == seed).all()
    for c in ['farm', 'day', 'hour']:
        assert np.array_equal(d[c], q[c])
    maxdiff(d.y, q.sub_ec)
    ref = outer[(outer.validator == v) & (outer.validation_fold == k) &
                (outer.seed == seed)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy()
    assert np.array_equal(ref, d.baseline.to_numpy())
    lo, hi = float(tr.sub_ec.min()), float(tr.sub_ec.max())
    finite(d[['baseline', 'candidate', 'new_et_raw', 'old_et_shrunk',
              'old_log_raw', 'seasonal_baseline']])
    assert (d.seasonal_baseline > 0).all()
    assert (d.baseline.between(lo, hi)).all() and (d.candidate.between(lo, hi)).all()
    bq = M.baseline(tr)(q)
    maxdiff(bq, d.seasonal_baseline)
    expected = np.clip(ref + .48 * (core.shrink(d.new_et_raw.to_numpy(), q)
                                    - d.old_et_shrunk.to_numpy()), lo, hi)
    scalar_gap = maxdiff(expected, d.candidate)
    assert (d.new_et_raw >= d.old_log_raw - 1e-10).all()
    return d, dict(sha256=S.sha(path), scalar_maxdiff=scalar_gap,
                   row_id_sha256=vector_id_sha(q.row_id), bounds=[lo, hi])

def fit_direction(core, a, b, cols, seed):
    lo, hi = float(a.sub_ec.min()), float(a.sub_ec.max())
    bm = M.baseline(a)
    btr, bq = bm(a), bm(b)
    ratio = a.sub_ec.to_numpy() / btr
    finite(btr); finite(bq); finite(ratio)
    assert (btr > 0).all() and (bq > 0).all() and (ratio > 0).all()
    old = core.et(seed); log = core.et(seed)
    for model in [old, log]:
        model.steps[-1][1].n_jobs = 2
    with threadpool_limits(limits=2):
        old.fit(a[cols], a.sub_ec)
        log.fit(a[cols], np.log(ratio))
    for model in [old, log]:
        model.steps[-1][1].n_jobs = 1
    counts, means = M.leaf_arrays(log, log.steps[0][1].transform(a[cols]), ratio)
    raw, old_log = M.predict(log, b, cols, bq, counts, means)
    old_raw = old.predict(b[cols])
    for value in [raw, old_log, old_raw]:
        finite(value)
    return old, log, counts, means, old_raw, raw, btr, bq, ratio

def weight(e, direction):
    assert len(e) == len(direction) and len(e) > 0
    finite(e); finite(direction)
    num = math.fsum(float(x) * float(y) for x, y in zip(e, direction)) / len(e)
    direction_mse = math.fsum(float(x) ** 2 for x in direction) / len(e)
    den = direction_mse + LAMBDA
    w = float(np.clip(-num / den, 0, 1))
    maxdiff([w], [np.clip(-np.mean(e * direction) / (np.mean(direction ** 2) + LAMBDA), 0, 1)])
    gradient = 2 * (num + den * w)
    assert ((w == 0 and gradient >= -1e-12) or
            (w == 1 and gradient <= 1e-12) or abs(gradient) < 1e-12)
    base_loss = math.fsum(float(x) ** 2 for x in e) / len(e)
    fit_loss = math.fsum(float(x) ** 2 for x in e + w * direction) / len(e)
    maxdiff([fit_loss], [base_loss + 2 * num * w + direction_mse * w * w])
    assert fit_loss + LAMBDA * w * w <= base_loss + 1e-12
    return w, dict(n=len(e), numerator=num, direction_mse=direction_mse,
                   denominator=den, gradient=gradient, weight=w,
                   baseline_mse=base_loss, fitted_mse=fit_loss,
                   penalized_fitted_loss=fit_loss + LAMBDA * w * w)

def check_first(core, a, b, cols, seed, old, log, counts, means, raw, old_raw, bq, ratio):
    replay = []
    for model, target, kind in [(old, a.sub_ec, 'old_et'), (log, np.log(ratio), 'log_et')]:
        fresh = clone(model); fresh.steps[-1][1].n_jobs = 2
        with threadpool_limits(limits=2):
            fresh.fit(a[cols], target)
        fresh.steps[-1][1].n_jobs = 1
        replay.append(dict(kind=kind, maxdiff=maxdiff(model.predict(b[cols]), fresh.predict(b[cols]))))
        if kind == 'log_et':
            cc, vv = M.leaf_arrays(fresh, fresh.steps[0][1].transform(a[cols]), ratio)
            rr, _ = M.predict(fresh, b, cols, bq, cc, vv)
            replay.append(dict(kind='log_ratio_leaf', maxdiff=maxdiff(rr, raw)))
        del fresh
    small, _ = M.predict(log, b.iloc[:8], cols, bq[:8], counts, means)
    batch = maxdiff(small, raw[:8])
    maxdiff(old.predict(b.iloc[:8][cols]), old_raw[:8])
    causal = []
    for farm in ['F13', 'F47']:
        for hour in [0, 6, 12]:
            keep = (b.farm == farm) & (b.hour <= hour)
            assert keep.any()
            changed = b.copy()
            changed.loc[~keep, cols] = changed.loc[~keep, cols] * 17 + 1000
            rr, _ = M.predict(log, changed, cols, bq, counts, means)
            oo = old.predict(changed[cols])
            maxdiff(rr[keep], raw[keep]); maxdiff(oo[keep], old_raw[keep])
            maxdiff(core.shrink(rr, b)[keep], core.shrink(raw, b)[keep])
            maxdiff(core.shrink(oo, b)[keep], core.shrink(old_raw, b)[keep])
            causal.append(dict(farm=farm, hour=hour, status='PASS'))
    manual = core.shrink(raw, b)
    for _, g in b.assign(p=raw).groupby(['farm', 'day']):
        history = []
        for j, row in g.sort_values('hour').iterrows():
            history.append(float(row.p))
            maxdiff([manual[j]], [.5 * row.p + .5 * math.fsum(history) / len(history)])
    return dict(status='PASS', replay=replay, batch_maxdiff=batch, causal=causal)

def validate_saved(dest, meta_path, inner_path, signature, ref, q):
    assert dest.exists() and meta_path.exists() and inner_path.exists(), 'partial cell; preserve and use new version'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    assert meta['signature'] == signature and meta['output_sha256'] == S.sha(dest)
    assert meta['inner_arrays_sha256'] == S.sha(inner_path)
    d = pd.read_csv(dest, float_precision='round_trip')
    assert np.array_equal(d.row_id, q.row_id) and d.row_id.is_unique
    for c in ['y', 'baseline', 'log_endpoint']:
        maxdiff(d[c], ref['candidate' if c == 'log_endpoint' else c])
    z = N.arrayfile(inner_path)
    assert vector_id_sha(z['train_row_id']) == signature['inner_train_ids']
    assert vector_id_sha(z['row_id']) == signature['inner_query_ids']
    assert vector_id_sha(z['train_y']) == signature['inner_targets']
    assert vector_id_sha(z['y']) == signature['inner_query_targets']
    assert (d.validator == signature['validator']).all()
    assert (d.fold == signature['fold']).all() and (d.seed == signature['seed']).all()
    for c in ['farm', 'day', 'hour']:
        assert np.array_equal(d[c], q[c])
    maxdiff(z['e'], z['baseline'] - z['y'])
    maxdiff(z['direction'], z['log_endpoint'] - z['baseline'])
    w, stats = weight(z['e'], z['direction'])
    for name, value in stats.items():
        maxdiff([value], [meta['weight_fit'][name]])
    maxdiff(d.weight, np.repeat(w, len(d)))
    maxdiff(d.direction, d.log_endpoint - d.baseline)
    maxdiff(d.candidate, d.baseline + w * d.direction)
    assert (d.candidate.between(*signature['bounds'])).all()
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    arg = parser.parse_args()
    lab, core, wv, folds, outer = S.loadec()
    assert len(folds) == 22
    check = N.preflight(lab, folds, outer)
    check.update(run_source_sha256=S.sha(Path(__file__)), log_source_sha256=M_SHA,
                 guard_source_sha256=N_SHA, support_source_sha256=S_SHA)
    assert (LOG / 'source_sha.txt').read_text() == M_SHA
    fit = json.loads((LOG_H / 'fit_audit_v1.json').read_text(encoding='utf-8'))
    assert fit['status'] == 'PASS' and len(fit['cells']) == 66
    assert {(x['validator'], x['fold'], x['seed']) for x in fit['cells']} == {
        (v, k, s) for v, k, _, _ in folds for s in SEEDS}
    assert json.loads((LOG_H / 'result_v1.json').read_text(encoding='utf-8'))['status'] == 'PASS'
    cols = [c for c in core.FULL if c != 'day'] + ['season']
    assert len(cols) == 38 and cols[-1] == 'season'
    references = {}
    for v, k, tm, vm in folds:
        tr, q = S.seasonal(lab[tm], lab[vm], wv)
        tr, q = tr.reset_index(drop=True), q.reset_index(drop=True)
        for seed in SEEDS:
            references[v, k, seed] = outer_reference(v, k, seed, tr, q, outer, core)
    check['outer_log_cells_audited'] = len(references)
    check['outer_log_cache_hashes'] = {f'{v}_{k}_{s}': m['sha256'] for (v, k, s), (_, m) in references.items()}
    if arg.prepare:
        dest = H / 'preparation_v1.json'
        S.savej(dest, check)
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return
    assert check['status'] == 'PASS', check
    OUT.mkdir(parents=True, exist_ok=True)
    source_path = OUT / 'source_sha.txt'
    source = S.sha(Path(__file__))
    if source_path.exists():
        assert source_path.read_text() == source
    else:
        source_path.write_text(source)
    idx = lab.set_index('row_id')
    audits = []
    for v, k, tm, vm in folds:
        a, b, z, bag = N.full_inner(v, k, lab[tm], idx)
        a, b = S.seasonal(a, b, wv)
        a, b = a.reset_index(drop=True), b.reset_index(drop=True)
        tr, q = S.seasonal(lab[tm], lab[vm], wv)
        tr, q = tr.reset_index(drop=True), q.reset_index(drop=True)
        for seed in SEEDS:
            ref, ref_meta = references[v, k, seed]
            stem = f'{v}_{k}_{seed}'
            dest, meta_path, inner_path = OUT / f'{stem}.csv', OUT / f'{stem}.json', OUT / f'{stem}_inner.npz'
            signature = dict(validator=v, fold=k, seed=seed, run_source_sha256=source,
                             inner_train_ids=vector_id_sha(a.row_id), inner_query_ids=vector_id_sha(b.row_id),
                             outer_train_ids=vector_id_sha(tr.row_id), outer_query_ids=vector_id_sha(q.row_id),
                             inner_targets=vector_id_sha(a.sub_ec), inner_query_targets=vector_id_sha(b.sub_ec),
                             outer_targets=vector_id_sha(q.sub_ec),
                             bounds=ref_meta['bounds'], log_cache_sha256=ref_meta['sha256'],
                             cpu_sha256=S.sha(N.INNER / f'{v}_{k}_cpu.npz'),
                             pfn_sha256=[S.sha(N.INNER / f'{v}_{k}_pfn_{s}.npz') for s in [1, 2, 3, 4]])
            if any(p.exists() for p in [dest, meta_path, inner_path]):
                audits.append(validate_saved(dest, meta_path, inner_path, signature, ref, q))
                print(v, k, seed, 'verified existing', flush=True)
                continue
            old, log, counts, means, old_raw, raw, btr, bq, ratio = fit_direction(core, a, b, cols, seed)
            lo, hi = float(a.sub_ec.min()), float(a.sub_ec.max())
            base = np.clip(core.shrink(.8 * z[f'r3_{seed}'] + .2 * bag, b), lo, hi)
            log_end, old_shrunk, new_shrunk, proposal = endpoint(core, base, raw, old_raw, b, lo, hi)
            e, direction = base - b.sub_ec.to_numpy(), log_end - base
            w, stats = weight(e, direction)
            if v == 'DIAG10' and k == 0 and seed == 7:
                first = check_first(core, a, b, cols, seed, old, log, counts, means, raw, old_raw, bq, ratio)
                S.savej(H / 'first_fold_verification_v1.json', dict(first, weight_fit=stats))
            baseline, endpoint_outer = ref.baseline.to_numpy(), ref.candidate.to_numpy()
            od = endpoint_outer - baseline
            pre = baseline + w * od
            pred = np.clip(pre, *ref_meta['bounds'])
            maxdiff(pre, pred)
            frame = q[['row_id', 'farm', 'day', 'hour']].copy()
            frame['y'] = q.sub_ec; frame['baseline'] = baseline; frame['log_endpoint'] = endpoint_outer
            frame['direction'] = od; frame['weight'] = w; frame['candidate'] = pred
            frame['validator'] = v; frame['fold'] = k; frame['seed'] = seed
            finite(frame[['y', 'baseline', 'candidate', 'log_endpoint', 'direction', 'weight']])
            np.savez(inner_path, train_row_id=a.row_id.to_numpy(str), row_id=b.row_id.to_numpy(str),
                     y=b.sub_ec.to_numpy(), train_y=a.sub_ec.to_numpy(), baseline=base,
                     log_endpoint=log_end, e=e, direction=direction, old_et_raw=old_raw,
                     log_raw=raw, old_et_shrunk=old_shrunk, log_shrunk=new_shrunk,
                     b_train=btr, b_query=bq, ratio_train=ratio, train_season=a.season.to_numpy(),
                     query_season=b.season.to_numpy(), inner_lo=lo, inner_hi=hi)
            frame.to_csv(dest, index=False)
            meta = dict(signature=signature, status='PASS', weight_fit=stats,
                        inner_train_support=support(a), inner_query_support=support(b),
                        outer_train_support=support(tr), outer_query_support=support(q),
                        inner_baseline_span=spans(a), outer_baseline_span=spans(tr),
                        inner_bounds=[lo, hi], inner_direction=quant(direction), outer_direction=quant(od),
                        inner_b_train=quant(btr), inner_b_query=quant(bq), inner_ratio=quant(ratio),
                        outer_b_query=quant(ref.seasonal_baseline),
                        inner_endpoint_clipped_rows=int(np.count_nonzero(proposal != log_end)),
                        outer_mix_clipped_rows=int(np.count_nonzero(pre != pred)),
                        outer_reference=ref_meta, output_sha256=S.sha(dest), inner_arrays_sha256=S.sha(inner_path))
            S.savej(meta_path, meta)
            audits.append(validate_saved(dest, meta_path, inner_path, signature, ref, q))
            print(v, k, seed, 'done weight', w, flush=True)
            del old, log, counts, means
            gc.collect()
    assert len(audits) == 66
    assert {(x['signature']['validator'], x['signature']['fold'], x['signature']['seed']) for x in audits} == set(references)
    S.savej(H / 'fit_audit_v1.json', dict(status='PASS', preflight=check, cells=audits))
    print('ALL_FITS_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
