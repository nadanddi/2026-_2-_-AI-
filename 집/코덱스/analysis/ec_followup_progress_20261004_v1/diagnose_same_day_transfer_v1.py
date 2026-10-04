"""Fixed completed public predictions only; descriptive, no model selection or fit.
Questions fixed in code: repeat weighting, within-day x movement, farm/phase mix.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import numpy as np
import pandas as pd

SRC = ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1/full_shift_records_v1.csv'
DEST = H/'same_day_transfer_v1.json'
assert not DEST.exists()
df = pd.read_csv(SRC, float_precision='round_trip')
assert len(df) == 31236
key = ['validator','fold','seed','hour','domain','farm','day']
assert not df.duplicated(key).any()
assert np.isfinite(df[['x','yday','required_shift','learned_correction']]).all().all()
assert np.max(np.abs(df.required_shift-(df.yday-df.x))) < 1e-12
check_count = len(df)

# Independent CSV/fsum equal-day aggregation rather than reusing pandas arrays.
manual = defaultdict(lambda: defaultdict(list))
with SRC.open(encoding='utf-8',newline='') as stream:
    for r in csv.DictReader(stream):
        if r['validator'] != 'DIAG10' or int(r['hour']) not in [0,23]:
            continue
        k = (int(r['seed']),int(r['hour']),r['domain'],r['farm'],int(r['day']))
        for name in ['x','yday','required_shift','learned_correction']:
            manual[k][name].append(float(r[name]))

sel = df[(df.validator=='DIAG10') & df.hour.isin([0,23])].copy()
daykey = ['seed','hour','domain','farm','day']
daily = sel.groupby(daykey,sort=True).agg(
    x=('x','mean'),yday=('yday','mean'),required_shift=('required_shift','mean'),
    learned_correction=('learned_correction','mean'),occurrences=('x','size')).reset_index()
for r in daily.itertuples(index=False):
    k = (r.seed,r.hour,r.domain,r.farm,r.day)
    for name in ['x','yday','required_shift','learned_correction']:
        assert abs(getattr(r,name)-math.fsum(manual[k][name])/len(manual[k][name])) < 1e-12
        check_count += 1
    assert r.occurrences == len(manual[k]['x'])
    assert max(manual[k]['yday'])-min(manual[k]['yday']) < 1e-12
    check_count += 2

def stats(g):
    if not len(g):
        return {'n':0}
    n = len(g)
    mean = lambda name: math.fsum(float(x) for x in g[name])/n
    return dict(n=n,unique_days=len(g[['farm','day']].drop_duplicates()),
                x=mean('x'),yday=mean('yday'),required_shift=mean('required_shift'),
                learned_correction=mean('learned_correction'),
                actual_high_n=int((g.yday>=1).sum()),
                actual_high_fraction=float((g.yday>=1).mean()))

repeat = []
strata = []
overlap = []
movement = []
paired_frames = []
for seed in [7,101,2024]:
    for hour in [0,23]:
        ss = sel[(sel.seed==seed)&(sel.hour==hour)]
        dd = daily[(daily.seed==seed)&(daily.hour==hour)]
        for domain in ['inner','outer']:
            occ = ss[(ss.domain==domain)&(ss.x>=.9)]
            unique = dd[(dd.domain==domain)&(dd.x>=.9)]
            # Selection is based on averaged x for equal-day comparison, stated explicitly.
            repeat.append(dict(seed=seed,hour=hour,domain=domain,
                               occurrence_selected=stats(occ),equal_day_mean_x_selected=stats(unique)))
            for farm in ['F13','F47']:
                for phase in ['early','late']:
                    g = unique[(unique.farm==farm)&((unique.day>=179) if phase=='late' else (unique.day<179))]
                    strata.append(dict(seed=seed,hour=hour,domain=domain,farm=farm,phase=phase,**stats(g)))
        inner = dd[dd.domain=='inner'].drop(columns='domain')
        outer = dd[dd.domain=='outer'].drop(columns='domain')
        assert len(outer)==360 and (outer.occurrences==1).all()
        pair = inner.merge(outer,on=['seed','hour','farm','day'],suffixes=('_inner','_outer'),validate='one_to_one')
        assert len(pair)==len(inner)
        assert np.max(np.abs(pair.yday_inner-pair.yday_outer)) < 1e-12
        pair['x_delta'] = pair.x_outer-pair.x_inner
        pair['needed_shift_delta'] = pair.required_shift_outer-pair.required_shift_inner
        assert np.max(np.abs(pair.x_delta+pair.needed_shift_delta)) < 1e-12
        check_count += 3*len(pair)
        ih,oh = pair.x_inner>=.9,pair.x_outer>=.9
        pair['membership'] = np.select([ih&oh,ih&~oh,~ih&oh],['both_high','inner_only_high','outer_only_high'],default='neither_high')
        paired_frames.append(pair)
        for group in ['all','actual_high','actual_ordinary','both_high','inner_only_high','outer_only_high','neither_high']:
            mask = np.ones(len(pair),dtype=bool) if group=='all' else ((pair.yday_inner>=1) if group=='actual_high' else ((pair.yday_inner<1) if group=='actual_ordinary' else pair.membership==group))
            g = pair[mask]
            rec = dict(seed=seed,hour=hour,group=group,n=len(g))
            if len(g):
                for name in ['x_inner','x_outer','x_delta','required_shift_inner','required_shift_outer','needed_shift_delta','learned_correction_inner','learned_correction_outer']:
                    rec[name] = float(g[name].mean())
                    assert abs(rec[name]-math.fsum(float(z) for z in g[name])/len(g)) < 1e-12
                    check_count += 1
                rec['x_delta_median'] = float(g.x_delta.median())
                rec['outer_prediction_higher_n'] = int((g.x_delta>0).sum())
            movement.append(rec)
        inner_set = {(r.farm,r.day) for r in inner[inner.x>=.9].itertuples()}
        outer_set = {(r.farm,r.day) for r in outer[outer.x>=.9].itertuples()}
        overlap.append(dict(seed=seed,hour=hour,inner_coverage=len(inner),outer_coverage=360,
                            inner_high=len(inner_set),outer_high=len(outer_set),
                            common=len(inner_set&outer_set),inner_only=len(inner_set-outer_set),
                            outer_only=len(outer_set-inner_set)))

pd.DataFrame(strata).to_csv(H/'same_day_transfer_strata_v1.csv',index=False)
pd.DataFrame(movement).to_csv(H/'same_day_transfer_movement_v1.csv',index=False)
pd.concat(paired_frames,ignore_index=True).to_csv(ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1/same_day_transfer_pairs_v1.csv',index=False)
result = dict(status='PASS',checks=check_count,input_rows=len(df),input_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              questions=['Does equal-day weighting retain the sign flip?',
                         'How does x move for the same farm-day in inner vs outer?',
                         'Do farm and pre-existing late phase strata share the flip?'],
              occurrence_vs_equal_day=repeat,overlap=overlap,movement=movement,strata=strata,
              scope='post-hoc diagnostic of one fixed completed public model; new fit/model selection zero',
              limitations=['Equal-day x averages several inner models and changes membership; it is a diagnostic, not a candidate prediction.',
                           'Inner coverage is selected and incomplete; matched subset is not representative of all 360 days.',
                           'Same-day comparison holds y fixed but not training set, feature transform, or model fit.',
                           'No inferential p-values, no claim of independent seeds or days, no causal attribution.',
                           'day>=179 uses the existing descriptive late segment; no new scoring subset chosen.',
                           'yday and actual high flags are diagnostic labels and never inference features.'])
DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=result['status'],checks=check_count,overlap=overlap,
                     selected=[m for m in movement if m['group'] in ['actual_high','both_high']]),ensure_ascii=False,indent=2))
