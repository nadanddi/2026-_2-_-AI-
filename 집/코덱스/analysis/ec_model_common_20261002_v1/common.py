"""읽기 전용 원자료/기준 캐시, 고정 분할, EC 5안 비교 공통 함수."""
from pathlib import Path
import sys, csv, json, hashlib, importlib.util
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '집' / '클로드' / 'research'))
import env
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

CORE_PATH = Path(env.CODEX) / 'rl_ec_v1' / 'run.py'
spec = importlib.util.spec_from_file_location('ec_readonly_common_core', CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
DATA = Path(env.DATA)
LOCK = Path(env.CODEX) / 'ec_final_lock' / 'locked_days.json'
SPLIT = ROOT / '집/코덱스/analysis/local/rl_ec_v1/20260927_173801/splits.csv'
CACHE = ROOT / '집/코덱스/local/ec_restart_phase3_20261001_v1'
RAW14 = ['out_temp', 'out_hum', 'out_rad', 'out_wspd'] + core.RAW
SEEDS = [7, 101, 2024]
VALIDATORS = ['DIAG10', 'A', 'B', 'EXT10', 'EXT12']
K = 5
ALPHA = .025 / K
NBOOT = 20000
BOOT_SEED = 918
_state = {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))

def near_mask(d, days):
    banned = {(f, v + j) for f, v in days for j in (-1, 0, 1)}
    return np.array([(f, int(v)) not in banned for f, v in zip(d.farm, d.day)])

def prepare():
    # Only MASK-permitted columns. Evaluation values are never loaded.
    raw = pd.read_csv(DATA / 'train_X.csv', usecols=['row_id'] + RAW14)
    raw = raw[raw.row_id.str[:3].isin(['F13', 'F47'])].reset_index(drop=True)
    full = core.identify(raw)
    locks = {(z['farm'], int(z['day'])) for z in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    records = []
    with (DATA / 'train_y.csv').open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            farm, day, _ = r['row_id'].split('_')
            if farm not in ('F13', 'F47') or (farm, int(day)) in locks:
                continue
            # Numeric conversion happens AFTER final-lock exclusion.
            records.append((r['row_id'], float(r['sub_ec'])))
    lab = core.features(raw).merge(pd.DataFrame(records, columns=['row_id', 'sub_ec']), on='row_id', validate='one_to_one')
    lab = lab.merge(pd.read_csv(SPLIT)[['row_id', 'fold', 'block']], on='row_id', validate='one_to_one')
    lab['late'] = lab.day.ge(179)
    assert len(lab) == 8640 and lab.row_id.is_unique
    assert len(lab[['farm', 'day']].drop_duplicates()) == 360
    assert not (set(lab[['farm', 'day']].itertuples(index=False, name=None)) & locks)
    temp = full.sort_values(['farm', 'day', 'hour']).copy()
    temp['smooth'] = temp.groupby('farm').in_temp.transform(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
    dmin = temp.groupby(['farm', 'day']).smooth.min()
    folds = [('DIAG10', i, {(f, int(d)) for f, d in lab.loc[lab.fold.eq(i), ['farm', 'day']].itertuples(index=False, name=None)}) for i in range(10)]
    layout = list(range(5)) + list(range(15, 25)) + list(range(35, 45)) + list(range(50, 55))
    for name, shifts in [('A', [70, 85, 100, 113, 126]), ('B', [63, 77, 92, 107, 120])]:
        for i, s in enumerate(shifts):
            folds.append((name, i, {(f, s + o) for f in ('F13', 'F47') for o in layout}))
    for th in (10, 12):
        folds.append((f'EXT{th}', 0, {(f, int(d)) for (f, d), v in dmin.items() if v < th and (f, int(d)) not in locks}))
    assert len(folds) == 22
    _state.update(lab=lab, locks=locks, folds=folds)
    return raw, lab, folds, locks, core

def split_fold(data, lab, fold, locks):
    name, i, days = fold
    va = lab[[(f, int(d)) in days for f, d in zip(lab.farm, lab.day)]].reset_index(drop=True)
    tr = lab[near_mask(lab, days | locks)].reset_index(drop=True)
    assert not (set(tr.row_id) & set(va.row_id))
    assert len(va) > 0 and len(tr) > 0
    assert not (set(tr[['farm', 'day']].itertuples(index=False, name=None)) & locks)
    for farm in ('F13', 'F47'):
        td = tr.loc[tr.farm.eq(farm), 'day'].unique()
        vd = va.loc[va.farm.eq(farm), 'day'].unique()
        if len(vd):
            assert np.min(np.abs(td[:, None] - vd[None, :])) >= 2
    return tr, va

def finish(p, tr, va):
    return np.clip(core.shrink(p, va), tr.sub_ec.min(), tr.sub_ec.max())

def inverse_shrink(va, q):
    a = va[['farm', 'day', 'hour']].reset_index(drop=True).copy()
    a['q'] = np.asarray(q, float)
    out = np.empty(len(a), dtype=float)
    for _, g in a.sort_values(['farm', 'day', 'hour']).groupby(['farm', 'day'], sort=False):
        total = 0.0
        for n, (ix, value) in enumerate(zip(g.index, g.q), 1):
            p = (2 * n * value - total) / (n + 1)
            out[ix] = p
            total += p
    assert np.max(np.abs(core.shrink(out, va) - np.asarray(q))) < 1e-12
    return out

def baseline(va, name, i, seed):
    path = CACHE / f'{name}_{i}.npz'
    with np.load(path) as z:
        assert z['row_id'].tolist() == va.row_id.tolist()
        r, v = z[f'r3_{seed}'].copy(), z[f'v2_{seed}'].copy()
    fold = next(f for f in _state['folds'] if f[:2] == (name, i))
    tr, _ = split_fold(None, _state['lab'], fold, _state['locks'])
    lo, hi = tr.sub_ec.min(), tr.sub_ec.max()
    cr = int(np.count_nonzero((r <= lo) | (r >= hi)))
    cv = int(np.count_nonzero((v <= lo) | (v >= hi)))
    return dict(r3=r, v2=v, pfn_finished=(v - .8 * r) / .2,
                cache_pfn_safe=(cr + cv == 0), cache_hash=sha(path),
                clip_r3_count=cr, clip_v2_count=cv)

def feature_checks(raw):
    old = core.features(raw).set_index('row_id')
    meta = core.identify(raw)
    pd.testing.assert_frame_equal(old, core.features(raw.sample(frac=1, random_state=61002)).set_index('row_id'))
    for farm in ('F13', 'F47'):
        day = int(meta.loc[meta.farm.eq(farm), 'day'].median())
        past = meta.farm.eq(farm) & (meta.day * 24 + meta.hour).le(day * 24 + 6)
        for mask in (meta.farm.eq(farm) & ~past, meta.farm.ne(farm)):
            new = raw.copy()
            new.loc[mask, RAW14] = new.loc[mask, RAW14] * 11 + 777
            pd.testing.assert_frame_equal(old.loc[meta.loc[past, 'row_id']], core.features(new).set_index('row_id').loc[meta.loc[past, 'row_id']])
        pd.testing.assert_frame_equal(old.loc[meta.loc[past, 'row_id']], core.features(raw[past]).set_index('row_id').loc[meta.loc[past, 'row_id']])
    new = raw.copy()
    new.loc[new.row_id.str.endswith('_00'), core.RAW] = np.nan
    assert core.features(new)[[c + '_h0' for c in core.RAW]].isna().all().all()
    return {'order_invariance': True, 'future_input_invariance': True,
            'other_farm_input_invariance': True, 'past_only_invariance': True,
            'missing_midnight_preserved': True, 'test_values_read': False,
            'final_lock_scored': False, 'mask_available_raw_columns': RAW14}

def evaluate(frame, arms):
    assert set(frame.seed) == set(SEEDS)
    assert set(frame.validator) == set(VALIDATORS)
    assert set(frame[['validator','seed']].itertuples(index=False,name=None)) == {(v,s) for v in VALIDATORS for s in SEEDS}
    expected_folds = {(v,i) for v,i,_ in _state['folds']}
    for seed in SEEDS:
        f = frame[frame.seed.eq(seed)]
        assert set(f[['validator','validation_fold']].itertuples(index=False,name=None)) == expected_folds
        for fold in _state['folds']:
            _, va = split_fold(None, _state['lab'], fold, _state['locks'])
            actual = f[f.validator.eq(fold[0]) & f.validation_fold.eq(fold[1])]
            assert set(actual.row_id) == set(va.row_id) and len(actual) == len(va)
    assert not frame.duplicated(['validator', 'validation_fold', 'seed', 'row_id']).any()
    assert np.isfinite(frame[['sub_ec', 'v2'] + arms].to_numpy()).all()
    scores, boots, decisions = [], [], []
    for (validator, seed), g in frame.groupby(['validator', 'seed'], sort=True):
        base = rmse(g.sub_ec, g.v2)
        folds = [rmse(d.sub_ec, d.v2) for _, d in g.groupby('validation_fold')]
        scores.append(dict(validator=validator, seed=int(seed), arm='v2', rmse=base,
                           fold_rmse_mean=float(np.mean(folds)), fold_rmse_std=float(np.std(folds, ddof=1)) if len(folds)>1 else None, n=len(g)))
        for arm in arms:
            cand = rmse(g.sub_ec, g[arm])
            ff = [rmse(d.sub_ec, d[arm]) for _, d in g.groupby('validation_fold')]
            scores.append(dict(validator=validator, seed=int(seed), arm=arm, rmse=cand,
                               delta_rmse=cand-base, delta_pct=100*(cand/base-1),
                               fold_rmse_mean=float(np.mean(ff)), fold_rmse_std=float(np.std(ff, ddof=1)) if len(ff)>1 else None, n=len(g)))
    for seed in SEEDS:
        g = frame[frame.validator.eq('DIAG10') & frame.seed.eq(seed)].copy()
        assert len(g) == 8640 and g.row_id.is_unique
        groups = g.groupby(['farm', 'block'], sort=True)
        n = groups.size().to_numpy(float)
        s0 = groups.apply(lambda x: float(((x.v2-x.sub_ec)**2).sum()), include_groups=False).to_numpy()
        assert len(n) == 80 and n.sum() == 8640
        rng = np.random.default_rng(BOOT_SEED)
        # Keep the original farm weights: independently resample fixed blocks within each farm.
        block_index = groups.size().index
        indices = []
        for farm in sorted(set(block_index.get_level_values('farm'))):
            choices = np.flatnonzero(block_index.get_level_values('farm') == farm)
            indices.append(choices[rng.integers(0, len(choices), (NBOOT, len(choices)))])
        idx = np.concatenate(indices, axis=1)
        den = n[idx].sum(axis=1)
        for arm in arms:
            sc = groups.apply(lambda x: float(((x[arm]-x.sub_ec)**2).sum()), include_groups=False).to_numpy()
            dm = (sc[idx]-s0[idx]).sum(axis=1)/den
            dr = np.sqrt(sc[idx].sum(axis=1)/den)-np.sqrt(s0[idx].sum(axis=1)/den)
            lo, hi = np.quantile(dm, [ALPHA, 1-ALPHA])
            boots.append(dict(seed=seed, arm=arm, n_blocks=len(n), n_rows=len(g), iterations=NBOOT,
                              bootstrap_seed=BOOT_SEED, family_k=K, alpha=ALPHA,
                              delta_mse=float((sc.sum()-s0.sum())/n.sum()),
                              ci_mse_low=float(lo), ci_mse_high=float(hi),
                              ci_rmse_low=float(np.quantile(dr, ALPHA)), ci_rmse_high=float(np.quantile(dr, 1-ALPHA)),
                              p_worse=float(np.mean(dm >= 0))))
    for arm in arms:
        ss = [s for s in scores if s['arm']==arm]
        bb = [b for b in boots if b['arm']==arm]
        assert len(ss) == 15 and len(bb) == 3
        direction = all(s['delta_rmse'] < 0 for s in ss)
        confident = all(b['p_worse'] < ALPHA and b['ci_mse_high'] < 0 for b in bb)
        decisions.append(dict(arm=arm, improving_seed_validator_cells=sum(s['delta_rmse']<0 for s in ss),
                              total_seed_validator_cells=15, direction_pass=direction,
                              diag_bootstrap_pass=confident, adopted=direction and confident,
                              label='검증 통과 후보' if direction and confident else '기각',
                              final_lock_scored=False, submission_created=False))
    # Ensemble-of-three-seeds summaries are diagnostics, not a separate arm or selection gate.
    avg = frame.groupby(['validator', 'validation_fold', 'row_id', 'farm', 'day', 'hour', 'block'], sort=True)[['sub_ec', 'v2']+arms].mean().reset_index()
    ensemble = []
    for name, g in avg.groupby('validator'):
        for arm in ['v2']+arms:
            ensemble.append(dict(validator=name, arm=arm, rmse=rmse(g.sub_ec, g[arm]), n=len(g)))
    return dict(scores=scores, bootstrap=boots, decisions=decisions, ensemble=ensemble,
                fixed_rule={'family_k':K,'alpha':ALPHA,'seed_validator_cells':15,'bootstrap_unit':'farm-stratified fixed 5-day block','n_bootstrap':NBOOT})

def manifest():
    import platform, sklearn, lightgbm
    return {'common_sha256':sha(__file__), 'core_sha256':sha(CORE_PATH), 'lock_sha256':sha(LOCK),
            'split_sha256':sha(SPLIT), 'input_sha256':{n:sha(DATA/n) for n in ('train_X.csv','train_y.csv')},
            'python':platform.python_version(), 'numpy':np.__version__, 'sklearn':sklearn.__version__,
            'lightgbm':lightgbm.__version__, 'seeds':SEEDS, 'final_lock_scored':False,
            'test_values_loaded':False}
