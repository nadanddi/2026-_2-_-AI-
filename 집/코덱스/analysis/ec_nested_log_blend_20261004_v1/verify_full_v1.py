"""Fresh public-only scoring and independent scalar/inner-fit audits."""
from pathlib import Path
import sys, json, math, argparse, importlib.util
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
sys.path.insert(0, str(ROOT / '집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear

def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m

N = load('full_verify_inner_ids', H.parent / 'ec_nested_high_specialist_20261004_v1/run_v2.py')
M = load('full_verify_log_b', H.parent / 'ec_log_partition_mean_20261004_v1/run.py')
checks = 0

def compare(a, b, tol=1e-12):
    global checks
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.size > 0
    assert np.isfinite(a).all() and np.isfinite(b).all()
    gap = float(np.max(np.abs(a - b)))
    assert gap < tol, gap
    checks += a.size
    return gap

def rms(y, p):
    value = math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(y, p)) / len(y))
    compare([value], [np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2))])
    return value

def basis(x):
    return np.column_stack([np.ones(len(x)), x, np.maximum(x - .6, 0), np.maximum(x - 1, 0)])

def boot_diag(d, seed, alpha):
    daily = {}
    for r in d.itertuples():
        daily.setdefault((r.farm, r.day), []).append((r.y - r.candidate) ** 2 - (r.y - r.baseline) ** 2)
    rng = np.random.default_rng(20261003 + seed)
    ss, nn = np.zeros(20000), np.zeros(20000)
    for farm in ['F13', 'F47']:
        days = sorted(day for f, day in daily if f == farm)
        sums, counts = [], []
        for i in range(0, len(days), 5):
            vals = [v for day in days[i:i + 5] for v in daily[farm, day]]
            sums.append(math.fsum(vals)); counts.append(len(vals))
        sums, counts = np.array(sums), np.array(counts)
        ix = rng.integers(len(sums), size=(20000, len(sums)))
        ss += sums[ix].sum(1); nn += counts[ix].sum(1)
    sample = ss / nn
    gd = d.assign(delta=(d.y-d.candidate)**2-(d.y-d.baseline)**2).groupby(['farm','day']).delta.agg(['sum','count'])
    rng = np.random.default_rng(20261003 + seed)
    ss2, nn2 = np.zeros(20000), np.zeros(20000)
    for farm in ['F13', 'F47']:
        g = gd.loc[farm].sort_index()
        sums = np.array([g['sum'].iloc[i:i+5].sum() for i in range(0,len(g),5)])
        counts = np.array([g['count'].iloc[i:i+5].sum() for i in range(0,len(g),5)])
        ix = rng.integers(len(sums), size=(20000,len(sums)))
        ss2 += sums[ix].sum(1); nn2 += counts[ix].sum(1)
    compare(sample, ss2/nn2)
    return dict(p_worse=float((sample >= 0).mean()), ci=np.quantile(sample,[alpha,1-alpha]).tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--recipe', choices=['matched','logblend'], required=True)
    parser.add_argument('--family', type=int, required=True)
    arg = parser.parse_args()
    eh = H.parent / arg.experiment
    out = ROOT / '집/코덱스/local' / arg.experiment
    result_path = eh / 'full_verification_v1.json'
    assert not result_path.exists()
    lab, core, wv, folds, outer = S.loadec()
    preflight = N.preflight(lab, folds, outer)
    assert preflight['status'] == 'PASS'
    if arg.recipe == 'matched':
        source_sha = N.EXPECTED
        paths = list(out.glob('*_pred.csv'))
        fit = json.loads((eh/'fit_audit_v1.json').read_text(encoding='utf-8'))
        coef = {(r['validator'],r['fold'],r['seed'],r['hour']): r for r in fit['coefficients']}
        assert len(coef) == 1584 and fit['checks'] == 1584
    else:
        source_sha = S.sha(eh/'run.py')
        paths = list(out.glob('*.csv'))
        fit = json.loads((eh/'fit_audit_v1.json').read_text(encoding='utf-8'))
        assert fit['status'] == 'PASS' and len(fit['cells']) == 66
    assert len(paths) == 66 and (out/'source_sha.txt').read_text() == source_sha
    frames = pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True)
    assert len(frames) == 83160 and not frames.duplicated(['validator','fold','seed','row_id']).any()
    expected = {(v,k,s,r) for v,k,tm,vm in folds for s in [7,101,2024] for r in lab.loc[vm,'row_id']}
    assert set(zip(frames.validator,frames.fold,frames.seed,frames.row_id)) == expected
    idx = lab.set_index('row_id')
    split_audit, inner_audit = [], []
    scalar_gap = 0
    for v,k,tm,vm in folds:
        tr, q = lab[tm].reset_index(drop=True),lab[vm].reset_index(drop=True)
        a,b,z,bag = N.full_inner(v,k,tr,idx)
        a,b = S.seasonal(a,b,wv); a,b = a.reset_index(drop=True),b.reset_index(drop=True)
        keys_tr = set(zip(tr.farm,tr.day)); keys_q = set(zip(q.farm,q.day))
        assert not keys_tr & keys_q
        assert all(not(f==ff and abs(day-qd)<=1) for f,day in keys_tr for ff,qd in keys_q)
        lo,hi = float(tr.sub_ec.min()),float(tr.sub_ec.max())
        split_audit.append(dict(validator=v,fold=k,train_days=len(keys_tr),query_days=len(keys_q)))
        for seed in [7,101,2024]:
            g = frames[(frames.validator==v)&(frames.fold==k)&(frames.seed==seed)].set_index('row_id').reindex(q.row_id).reset_index()
            compare(g.y,q.sub_ec)
            baseline = outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy()
            assert np.array_equal(g.baseline.to_numpy(),baseline)
            if arg.recipe == 'matched':
                inner_base = np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),float(z['lo']),float(z['hi']))
                yday = b.groupby(['farm','day']).sub_ec.mean()
                for hour in range(24):
                    rec = coef[v,k,seed,hour]
                    bm = b.hour <= hour
                    x = b.loc[bm,['farm','day']].assign(p=inner_base[bm]).groupby(['farm','day']).p.mean()
                    assert rec['n_days'] == len(x)
                    target = yday.reindex(x.index).to_numpy() - x.to_numpy()
                    design = np.vstack([basis(x.to_numpy()),np.sqrt(10)*np.eye(4)])
                    target_full = np.r_[target,np.zeros(4)]
                    beta = np.asarray(rec['beta'])
                    replay = lsq_linear(design,target_full,bounds=([-np.inf,-1,0,0],[np.inf,np.inf,np.inf,np.inf]),tol=1e-12,max_iter=300)
                    assert replay.success
                    gap = compare(beta,replay.x,1e-8)
                    grad = design.T @ (design @ beta - target_full)
                    lower = np.array([-np.inf,-1,0,0])
                    active = np.isfinite(lower)&(beta-lower<1e-6)
                    assert np.max(np.abs(grad[~active])) < 1e-6
                    assert not active.any() or np.min(grad[active]) > -1e-6
                    inner_audit.append(dict(validator=v,fold=k,seed=seed,hour=hour,coef_maxdiff=gap))
                for _,dayg in g.groupby(['farm','day']):
                    history = []
                    for r in dayg.sort_values('hour').itertuples():
                        history.append(float(r.baseline))
                        x = math.fsum(history)/len(history)
                        beta = coef[v,k,seed,r.hour]['beta']
                        value = math.fsum(float(t)*float(c) for t,c in zip([1,x,max(x-.6,0),max(x-1,0)],beta))
                        correction = .2*min(.3,max(-.3,value))
                        scalar_gap = max(scalar_gap,compare([correction],[r.correction]))
                        scalar_gap = max(scalar_gap,compare([min(hi,max(lo,r.baseline+correction))],[r.candidate]))
            else:
                stem = f'{v}_{k}_{seed}'
                path = out / f'{stem}.csv'; npz = out / f'{stem}_inner.npz'
                meta = json.loads((out/f'{stem}.json').read_text(encoding='utf-8'))
                assert meta['output_sha256'] == S.sha(path) and meta['inner_arrays_sha256'] == S.sha(npz)
                saved = N.arrayfile(npz)
                assert np.array_equal(saved['train_row_id'],a.row_id) and np.array_equal(saved['row_id'],b.row_id)
                compare(saved['train_y'],a.sub_ec); compare(saved['y'],b.sub_ec)
                bm = M.baseline(a)
                compare(saved['b_train'],bm(a)); compare(saved['b_query'],bm(b))
                compare(saved['ratio_train'],a.sub_ec.to_numpy()/saved['b_train'])
                ib = np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),float(z['lo']),float(z['hi']))
                compare(saved['baseline'],ib)
                endpoint = np.clip(ib+.48*(core.shrink(saved['log_raw'],b)-core.shrink(saved['old_et_raw'],b)),float(z['lo']),float(z['hi']))
                compare(saved['log_endpoint'],endpoint)
                e, direction = ib-b.sub_ec.to_numpy(),endpoint-ib
                compare(saved['e'],e); compare(saved['direction'],direction)
                num = math.fsum(float(x)*float(y) for x,y in zip(e,direction))/len(e)
                den = math.fsum(float(x)**2 for x in direction)/len(e)+.01
                w = min(1,max(0,-num/den))
                compare([num,den,w],[meta['weight_fit']['numerator'],meta['weight_fit']['denominator'],meta['weight_fit']['weight']])
                grad = 2*(num+den*w)
                assert (w==0 and grad>=-1e-12) or (w==1 and grad<=1e-12) or abs(grad)<1e-12
                prior = pd.read_csv(M.OUT/f'{stem}.csv',float_precision='round_trip')
                assert np.array_equal(prior.row_id,q.row_id)
                compare(g.log_endpoint,prior.candidate)
                compare(g.weight,np.repeat(w,len(g)))
                expected_p = baseline+w*(prior.candidate.to_numpy()-baseline)
                compare(expected_p,np.clip(expected_p,lo,hi))
                scalar_gap = max(scalar_gap,compare(g.candidate,expected_p))
                inner_audit.append(dict(validator=v,fold=k,seed=seed,weight=w,gradient=grad,n_query=len(b),
                                        inner_baseline_mse=float(np.mean(e**2)),inner_fitted_mse=float(np.mean((e+w*direction)**2))))
    scores = []
    for (v,seed),g in frames.groupby(['validator','seed']):
        rb,rc = rms(g.y,g.baseline),rms(g.y,g.candidate)
        scores.append(dict(validator=v,seed=int(seed),n=len(g),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1)))
    scores = pd.DataFrame(scores)
    alpha = .025/arg.family
    boots = {seed:boot_diag(frames[(frames.validator=='DIAG10')&(frames.seed==seed)],seed,alpha) for seed in [7,101,2024]}
    segments = []
    for seed in [7,101,2024]:
        d = frames[(frames.validator=='DIAG10')&(frames.seed==seed)]
        high = d.groupby(['farm','day']).y.transform('mean')>=1
        for label,mask in [('high',high),('ordinary',~high),('late',d.day>=179),('F13',d.farm=='F13'),('F47',d.farm=='F47'),('hour0',d.hour==0)]:
            g = d[mask]
            segments.append(dict(seed=seed,segment=label,n=len(g),days=len(g[['farm','day']].drop_duplicates()),
                                 baseline=rms(g.y,g.baseline),candidate=rms(g.y,g.candidate),
                                 baseline_bias=float((g.baseline-g.y).mean()),candidate_bias=float((g.candidate-g.y).mean())))
    passed = bool((scores.change_pct<0).all() and all(b['p_worse']<alpha and b['ci'][1]<0 for b in boots.values()))
    scores.to_csv(eh/'full_scores_v1.csv',index=False)
    pd.DataFrame(segments).to_csv(eh/'full_segments_v1.csv',index=False)
    result = dict(status='PASS',decision='PUBLIC_PASS_PENDING_FURTHER_GUARD' if passed else 'REJECT',public_pass=passed,
                  recipe=arg.recipe,family=arg.family,alpha=alpha,rows=len(frames),checks=checks,
                  bootstrap=boots,scalar_maxdiff=scalar_gap,inner_audit=inner_audit,split_audit=split_audit,
                  source_sha256=source_sha,verifier_sha256=S.sha(Path(__file__)),preflight=preflight,
                  limitations=['repeated public validation, not untouched holdout','no EL1/test/new lock scoring','inner fitting gains are not independent performance'])
    S.savej(result_path,result)
    print(scores.to_string(index=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['inner_audit','split_audit','preflight']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
