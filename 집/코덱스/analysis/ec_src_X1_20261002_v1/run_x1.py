from pathlib import Path
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'집/클로드/research'))
import env
import csv, json, hashlib, math, time
from collections import defaultdict, Counter
import numpy as np

HERE = Path(__file__).resolve().parent
FAMILIES = ['identity', 'offset', 'scale', 'affine']
SHIFTS = list(range(-6, 7))
REPS = 999
SEED = 20261002

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)

def writecsv(name, data):
    data = list(data)
    if not data:
        (HERE/name).write_text('', encoding='utf-8')
        return
    with (HERE/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)

def load():
    lockpath = Path(env.CODEX)/'ec_final_lock/locked_days.json'
    locks = {(r['farm'], int(r['day'])) for r in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
    d = defaultdict(dict)
    truth = {}
    presence = defaultdict(Counter)
    excluded = 0
    for r in rows(Path(env.DATA)/'train_y.csv'):
        f, day, hour = r['row_id'].split('_')
        day, hour = int(day), int(hour)
        # String inventory only outside the two target farms. Locks before float.
        present = r['sub_ec'].strip().lower() not in ['', 'nan', 'na', 'null']
        presence[f]['rows'] += 1
        presence[f]['ec_nonempty'] += int(present)
        if f not in ['F13', 'F47']:
            continue
        if (f, day) in locks:
            excluded += 1
            continue
        if present:
            assert hour not in d[(f, day)]
            d[(f, day)][hour] = float(r['sub_ec'])
            truth[r['row_id']] = d[(f, day)][hour]
    assert excluded == 40*24
    complete = sorted(k for k, v in d.items() if set(v) == set(range(24)))
    assert len(complete) == len(d), 'Incomplete unlocked EC day'
    y = np.array([[d[k][h] for h in range(24)] for k in complete], dtype=np.float64)
    e = defaultdict(list)
    row_err = []
    for r in rows(ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1/oof_predictions.csv'):
        if r['validator'] != 'DIAG10' or r['row_id'] not in truth:
            continue
        assert abs(float(r['sub_ec'])-truth[r['row_id']]) < 1e-12
        f, day, _ = r['row_id'].split('_')
        error = float(r['v2'])-truth[r['row_id']]
        e[(f, int(day))].append(error)
        row_err.append(error)
    assert set(e) == set(complete) and all(len(v) == 24 for v in e.values())
    level_rmse = math.sqrt(math.fsum((math.fsum(v)/24)**2 for v in e.values())/len(e))
    row_rmse = math.sqrt(math.fsum(x*x for x in row_err)/len(row_err))
    strata = []
    for f in ['F13', 'F47']:
        inds = np.array([i for i,k in enumerate(complete) if k[0] == f])
        for b, selected in enumerate(np.array_split(inds, 3)):
            strata.append((f'{f}_B{b+1}', selected))
    return complete, y, strata, level_rmse, row_rmse, dict(presence), excluded

def informative(y):
    diff = np.diff(y, axis=1)
    unique = np.array([len(np.unique(np.rint(z/.001))) for z in diff])
    return ((np.ptp(y, axis=1) >= .03) & (y.std(axis=1) >= .008)
            & ((np.abs(diff) >= .000999).sum(axis=1) >= 8) & (unique >= 6))

def pairsearch(y, keys, need_level, threshold, collect=False, cross=False):
    n = len(y)
    means = y.mean(axis=1)
    good = informative(y)
    leveldiff = np.abs(means[:,None]-means[None,:])
    allowed = np.triu(np.ones((n,n), dtype=bool), 1)
    days = np.array([k[1] for k in keys])
    farms = np.array([k[0] for k in keys])
    same = farms[:,None] == farms[None,:]
    allowed &= (~same | (np.abs(days[:,None]-days[None,:]) >= 7))
    if need_level:
        allowed &= leveldiff <= threshold
    allowed &= good[:,None] & good[None,:]
    best = {f: {} for f in FAMILIES}
    for shift in SHIFTS:
        # y_i(t) = a y_j(t+shift) + b on the non-wrapping overlap.
        start = max(0,-shift)
        end = min(24,24-shift)
        target = y[:, start:end]
        source = y[:, start+shift:end+shift]
        m = end-start
        mt = target.mean(axis=1)
        ms = source.mean(axis=1)
        st2 = np.mean(target*target, axis=1)
        ss2 = np.mean(source*source, axis=1)
        dot = target@source.T/m
        var_s = np.maximum(ss2-ms*ms, 1e-20)
        cov = dot-mt[:,None]*ms[None,:]
        for family in FAMILIES:
            if family == 'identity':
                a = np.ones((n,n)); b = np.zeros((n,n))
            elif family == 'offset':
                a = np.ones((n,n)); b = mt[:,None]-ms[None,:]
            elif family == 'scale':
                a = dot/np.maximum(ss2[None,:], 1e-20); b = np.zeros((n,n))
            else:
                a = cov/var_s[None,:]; b = mt[:,None]-a*ms[None,:]
            mse = st2[:,None] + a*a*ss2[None,:] + b*b - 2*a*dot - 2*b*mt[:,None] + 2*a*b*ms[None,:]
            candidates = allowed & (a >= .5) & (a <= 2) & (mse <= .0005**2+1e-14)
            for i,j in zip(*np.where(candidates)):
                err = target[i]-(a[i,j]*source[j]+b[i,j])
                maxabs = float(np.max(np.abs(err)))
                if maxabs > .001001:
                    continue
                rmse = float(np.sqrt(np.mean(err*err)))
                old = best[family].get((int(i),int(j)))
                if old is None or rmse < old['rmse']:
                    best[family][(int(i),int(j))] = {
                        'family':family, 'farm_i':keys[i][0], 'day_i':keys[i][1],
                        'farm_j':keys[j][0], 'day_j':keys[j][1], 'shift':shift,
                        'a':float(a[i,j]), 'b':float(b[i,j]), 'overlap':m,
                        'rmse':rmse, 'maxabs':maxabs, 'level_difference':float(leveldiff[i,j])}
    counts = np.zeros(4, dtype=int)
    selected_all = []
    allpairs = []
    for fi, family in enumerate(FAMILIES):
        used = []
        candidates = sorted(best[family].items(), key=lambda p:(p[1]['rmse'],p[0]))
        for (i,j), p in candidates:
            independent = all(not (keys[z][0] == keys[u][0] and abs(keys[z][1]-keys[u][1])<7)
                              for z in (i,j) for u in used)
            if independent:
                counts[fi] += 1
                used.extend([i,j])
                if collect:
                    selected_all.append(dict(p))
            if collect:
                allpairs.append(dict(p))
    return counts, selected_all, allpairs

def iaaft(y, rng):
    # All original values retained exactly; 20 alternating spectrum/rank passes.
    sorted_y = np.sort(y, axis=1)
    amps = np.abs(np.fft.rfft(y, axis=1))
    z = np.take_along_axis(y, np.argsort(rng.random(y.shape), axis=1), axis=1)
    for _ in range(20):
        zf = np.fft.rfft(z, axis=1)
        unit = zf/np.maximum(np.abs(zf), 1e-30)
        x = np.fft.irfft(amps*unit, n=24, axis=1)
        ranks = np.argsort(np.argsort(x, axis=1), axis=1)
        z = np.take_along_axis(sorted_y, ranks, axis=1)
    original_non_dc = amps[:,1:]
    transformed = np.abs(np.fft.rfft(z,axis=1))[:,1:]
    rel = np.sqrt(np.sum((transformed-original_non_dc)**2,axis=1))/np.maximum(np.sqrt(np.sum(original_non_dc**2,axis=1)),1e-20)
    assert np.array_equal(np.sort(z,axis=1),sorted_y)
    return z, rel

def flat_inventory(keys,y):
    runs=[]
    signatures=defaultdict(list)
    for k, row in zip(keys,y):
        rounded=np.rint(row/.001).astype(int)
        start=0
        for h in range(1,25):
            if h==24 or rounded[h]!=rounded[start]:
                if h-start>=3:
                    runs.append({'farm':k[0],'day':k[1],'start_hour':start,'length':h-start,'value':rounded[start]/1000})
                start=h
        for h in range(13):
            window=rounded[h:h+12]
            diffs=tuple(np.diff(window).tolist())
            signatures[diffs].append((k[0],k[1],h,int(np.ptp(window))))
    repeated=[]
    for sig, locations in signatures.items():
        unique_days={(v[0],v[1]) for v in locations}
        if len(unique_days)>=2:
            repeated.append({'signature':';'.join(str(x) for x in sig),'occurrences':len(locations),
                             'unique_days':len(unique_days),'range_min':min(v[3] for v in locations)/1000,
                             'range_max':max(v[3] for v in locations)/1000,
                             'flat':max(abs(x) for x in sig)==0,
                             'locations':';'.join(f'{f}_{d}_{h}' for f,d,h,_ in locations[:20])})
    return runs, sorted(repeated,key=lambda z:(-z['unique_days'],z['signature']))

def main():
    t=time.monotonic()
    keys,y,strata,baseline_level,baseline_row,presence,excluded=load()
    threshold=baseline_level/4
    print(json.dumps({'stage':'loaded','days':len(keys),'rows':y.size,'baseline_level_rmse':baseline_level,'level_diff_limit':threshold},ensure_ascii=False),flush=True)
    observed=np.zeros((6,4),dtype=int)
    selected=[]; allpairs=[]
    for si,(name,inds) in enumerate(strata):
        k=[keys[i] for i in inds]
        c,p,a=pairsearch(y[inds],k,True,threshold,True)
        observed[si]=c
        for collection,values in [(selected,p),(allpairs,a)]:
            for v in values: collection.append({'stratum':name,**v})
    print(json.dumps({'stage':'observed','counts':observed.tolist()},ensure_ascii=False),flush=True)
    rng=np.random.default_rng(SEED)
    null=np.zeros((REPS,6,4),dtype=int)
    spectral=[]
    for b in range(REPS):
        surrogate,rel=iaaft(y,rng)
        spectral.extend(rel.tolist())
        for si,(name,inds) in enumerate(strata):
            null[b,si]=pairsearch(surrogate[inds],[keys[i] for i in inds],True,threshold)[0]
        if (b+1)%25==0:
            print(json.dumps({'stage':'null','done':b+1,'reps':REPS,'seconds':round(time.monotonic()-t,2)},ensure_ascii=False),flush=True)
        if time.monotonic()-t>1800:
            raise RuntimeError('Precommitted 30-minute budget exceeded; incomplete, no success declaration')
    totals=observed.sum(axis=0)
    p=((null.sum(axis=1)>=totals).sum(axis=0)+1)/(REPS+1)
    stratum_p=((null>=observed[None,:,:]).sum(axis=0)+1)/(REPS+1)
    discovery=(p<.0125)&(observed.min(axis=0)>=1)
    # Descriptive all-day/cross-farm scope, exact same fixed search, no extra tests.
    c,unused,global_pairs=pairsearch(y,keys,False,threshold,True)
    informative_days=informative(y)
    runs, signatures=flat_inventory(keys,y)
    dayrecords=[]
    for i,k in enumerate(keys):
        b=next(name for name,inds in strata if i in inds)
        dayrecords.append({'farm':k[0],'day':k[1],'stratum':b,'mean_ec':float(y[i].mean()),'range_ec':float(np.ptp(y[i])),
                           'std_ec':float(y[i].std()),'informative':bool(informative_days[i])})
    writecsv('days.csv',dayrecords)
    writecsv('selected_pairs.csv',selected)
    writecsv('all_primary_pairs.csv',allpairs)
    writecsv('all_scope_shape_matches.csv',global_pairs)
    writecsv('plateau_runs.csv',runs)
    writecsv('repeated_12h_diff_signatures.csv',signatures)
    writecsv('null_counts.csv',[{'rep':b+1,**{f'{name}/{family}':int(null[b,si,fi]) for si,(name,_) in enumerate(strata) for fi,family in enumerate(FAMILIES)}} for b in range(REPS)])
    examples=[]
    for pi,pair in enumerate(sorted(allpairs,key=lambda q:q['rmse'])[:8]):
        i=keys.index((pair['farm_i'],pair['day_i']));j=keys.index((pair['farm_j'],pair['day_j']))
        for h in range(24):
            sh=h+pair['shift']
            examples.append({'pair_index':pi, 'family':pair['family'],'farm_i':keys[i][0],'day_i':keys[i][1],
                             'farm_j':keys[j][0],'day_j':keys[j][1],'hour_i':h,'hour_j':sh,
                             'ec_i':float(y[i,h]),'ec_j_shifted':float(y[j,sh]) if 0<=sh<24 else '',
                             'transformed_ec_j':float(pair['a']*y[j,sh]+pair['b']) if 0<=sh<24 else ''})
    writecsv('raw_pair_examples.csv',examples)
    summary={'status':'COMPLETE','seed':SEED,'null_reps':REPS,'families':FAMILIES,'shifts':SHIFTS,
             'days':len(keys),'rows':int(y.size),'lock_days_excluded':40,'lock_rows_excluded_before_float':excluded,
             'otherfarm_label_inventory':presence,'v2_diag10_row_rmse':baseline_row,'v2_day_level_rmse':baseline_level,
             'level_difference_limit':threshold,'informative_days':int(informative_days.sum()),
             'strata':[name for name,_ in strata],'observed_independent_pairs':observed.tolist(),
             'observed_total_pairs':totals.tolist(),'p_per_family':p.tolist(),'p_bonferroni':np.minimum(1,p*4).tolist(),
             'p_per_stratum_unadjusted':stratum_p.tolist(),'discovery_per_family':discovery.tolist(),
             'all_primary_pair_count':len(allpairs),'all_scope_shape_match_count':len(global_pairs),
             'all_scope_smalllevel_match_count':sum(z['level_difference']<=threshold for z in global_pairs),
             'plateau_runs_ge3h':len(runs),'repeated_12h_diff_signatures':len(signatures),
             'repeated_12h_flat_signatures':sum(z['flat'] for z in signatures),
             'repeated_12h_nonflat_signatures':sum(not z['flat'] for z in signatures),
             'null_spectral_relative_error_median':float(np.median(spectral)),
             'null_spectral_relative_error_p95':float(np.quantile(spectral,.95)),
             'elapsed_seconds':time.monotonic()-t,'model_training':False,'evaluation_features_available':False,
             'hashes':{str(q.relative_to(ROOT)) if q.is_relative_to(ROOT) else str(q):sha(q) for q in [Path(__file__),HERE/'PROTOCOL.md',Path(env.DATA)/'train_y.csv',Path(env.CODEX)/'ec_final_lock/locked_days.json',ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1/oof_predictions.csv']}}
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
