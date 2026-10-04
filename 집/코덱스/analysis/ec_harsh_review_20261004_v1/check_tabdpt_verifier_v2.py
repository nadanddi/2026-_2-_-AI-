"""Review verifier v1 using only source AST and synthetic arrays.

Do not import the verifier, fit/predict a model, or read any actual result/data.
"""
import ast
import copy
import hashlib
import json
import math
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path('C:/work/farmai')
HERE = ROOT / '집/코덱스/analysis/ec_harsh_review_20261004_v1'
EXP = ROOT / '집/코덱스/analysis/ec_tabdpt_20261004_v1'
RESULT = HERE / 'tabdpt_verifier_check_v1.json'
assert not RESULT.exists() and __debug__
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import numpy as np
import pandas as pd

source = EXP / 'verify_full_v1.py'
tree = ast.parse(source.read_text(encoding='utf-8-sig'))
run = ast.parse((EXP / 'run_v4.py').read_text(encoding='utf-8-sig'))
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
namespace = dict(np=np, pd=pd, math=math, CHECKS=0,
                 SEEDS=(7, 101, 2024), VALIDATORS=('DIAG10', 'A', 'B', 'EXT10', 'EXT12'),
                 ALPHA=.025/20, RAW_ATOL=1e-6, FINAL_ATOL=2e-7)
selected = [copy.deepcopy(functions[n]) for n in
            ['compare', 'rmse_both', 'prefix_scalar', 'boot_diag', 'validate_first', 'calculate_scores']]
exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
             '<actual-verifier-functions>', 'exec'), namespace)
compare = namespace['compare']
boot = namespace['boot_diag']
prefix = namespace['prefix_scalar']
records = []


def good(name, **extra):
    records.append(dict(case=name, status='PASS', **extra))


def rejects(name, fn):
    try:
        fn()
    except (AssertionError, KeyError):
        good(name)
    else:
        raise AssertionError('Invalid synthetic record accepted: ' + name)


# Independent exact rational prefix averages, intentionally shuffled and clipped.
frame = pd.DataFrame(dict(farm=['F47','F13','F13','F47','F13'],
                          day=[2,1,1,2,1], hour=[1,2,0,0,1]))
r3 = np.array([5.,2.,1.,0.,0.]); new = np.array([-5.,6.,0.,0.,10.])
exact = []
for i, row in frame.iterrows():
    indices = [j for j, x in frame.iterrows()
               if x.farm == row.farm and x.day == row.day and x.hour <= row.hour]
    raw = lambda j: Fraction(4,5)*Fraction(float(r3[j]))+Fraction(1,5)*Fraction(float(new[j]))
    value = (raw(i)+sum((raw(j) for j in indices), Fraction(0))/len(indices))/2
    exact.append(float(min(Fraction(2), max(Fraction(1,10), value))))
compare(prefix(frame, r3, new, .1, 2.), exact)
core_tree = ast.parse((ROOT/'집/코덱스/analysis/codex_independent/rl_ec_v1/run.py').read_text(encoding='utf-8-sig'))
shrink_node = next(n for n in core_tree.body if isinstance(n,ast.FunctionDef) and n.name=='shrink')
core_ns = dict(np=np, pd=pd)
exec(compile(ast.Module(body=[shrink_node], type_ignores=[]), '<actual-core-shrink>', 'exec'), core_ns)
compare(prefix(frame,r3,new,.1,2.), np.clip(core_ns['shrink'](.8*r3+.2*new,frame),.1,2.))
good('independent_fraction_prefix_and_actual_shrink', values=exact)
duplicate = pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
rejects('prefix_duplicate_hour', lambda: prefix(duplicate, np.append(r3,r3[0]),np.append(new,new[0]),.1,2.))
rejects('prefix_nonfinite_raw', lambda: prefix(frame, np.array([np.inf,2,1,0,0]),new,.1,2.))

# Variable errors and variable row counts detect block weighting and the short final block.
toy = []
for farm_index, farm in enumerate(['F13','F47']):
    for day in range(13):
        for hour in range(2+day%4):
            y = .6+hour/100
            old_error = .1+(day%3)/10
            new_error = .08+(day//5)/20+farm_index/30
            toy.append(dict(farm=farm, day=day, hour=hour, y=y,
                            baseline=y+old_error, candidate=y+new_error))
toy = pd.DataFrame(toy)
actual_boot = boot(toy,101)
blocks = {}
for farm in ['F13','F47']:
    days = sorted(set(toy.loc[toy.farm==farm,'day']))
    farm_blocks = []
    for start in range(0,len(days),5):
        selected = toy[(toy.farm==farm)&toy.day.isin(days[start:start+5])]
        terms = []
        for row in selected.itertuples():
            y,b,c = map(Decimal.from_float,[float(row.y),float(row.baseline),float(row.candidate)])
            terms.append((y-c)**2-(y-b)**2)
        farm_blocks.append((float(sum(terms,Decimal(0))),len(selected)))
    blocks[farm] = farm_blocks
rng = np.random.default_rng(20261003+101)
draws = {f:rng.integers(len(blocks[f]),size=(20000,len(blocks[f]))) for f in ['F13','F47']}
samples = []
for iteration in range(20000):
    sums, counts = [], []
    for f in ['F13','F47']:
        for chosen in draws[f][iteration]:
            value,count = blocks[f][int(chosen)]; sums.append(value);counts.append(count)
    samples.append(math.fsum(sums)/sum(counts))
samples = np.asarray(samples)
independent_p = float(np.mean(samples>=0))
independent_ci = np.quantile(samples,[.00125,.99875])
assert actual_boot['p_worse']==independent_p
compare(actual_boot['ci_family20'], independent_ci)
assert actual_boot['blocks_per_farm']==dict(F13=3,F47=3)
assert actual_boot['adjusted_quantiles']==[.00125,.99875]
good('decimal_and_scalar_bootstrap_nonconstant_variable_count', p_worse=independent_p,
     ci=independent_ci.tolist(), blocks_per_farm=actual_boot['blocks_per_farm'])
shuffled = boot(toy.sample(frac=1,random_state=321),101)
assert shuffled['p_worse']==actual_boot['p_worse']
compare(shuffled['ci_family20'],actual_boot['ci_family20'])
good('bootstrap_input_order_invariance')
tied = toy.copy();tied['candidate']=tied.baseline
tie_boot = boot(tied,7)
assert tie_boot['p_worse']==1 and tie_boot['ci_family20']==[0.,0.]
good('bootstrap_ties_count_as_worse')
rejects('bootstrap_duplicate_hour', lambda: boot(pd.concat([toy,toy.iloc[[0]]]),7))

# Actual 15-cell function on 100% synthetic 360-day/24-hour DIAG support.
rows=[]
for farm in ['F13','F47']:
    for day in range(180):
        y=1. if farm=='F13' and day<31 else .5
        for hour in range(24):
            rows.append(dict(farm=farm,day=day,hour=hour,y=y,baseline=y+.2,candidate=y+.1))
dayframe=pd.DataFrame(rows)
full=pd.concat([dayframe.assign(validator=v,seed=seed) for v in namespace['VALIDATORS']
                for seed in namespace['SEEDS']],ignore_index=True)
scores,segments,boots,passed=namespace['calculate_scores'](full)
assert passed and len(scores)==15 and len(segments)==18
assert set(segments[segments.segment=='high'].days)=={31}
assert set(segments[segments.segment=='ordinary'].days)=={329}
assert set(segments[segments.segment=='high'].n)=={744}
assert set(segments[segments.segment=='ordinary'].n)=={7896}
assert all(b['p_worse']==0 and b['ci_family20'][1]<0 for b in boots.values())
compare(scores.baseline,np.repeat(.2,15));compare(scores.candidate,np.repeat(.1,15))
good('full_15_cell_scoring_and_inclusive_high_31', score_cells=15,
     diag_rows_per_seed=8640, high_days=31, ordinary_days=329,
     all_scores_negative=True, public_pass=passed)

# Test the exact final decision expression separately, including strict boundaries.
pass_node = next(n for n in functions['calculate_scores'].body
                 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='passed' for t in n.targets))
pass_code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(pass_node)],type_ignores=[])),
                  '<actual-final-decision>', 'exec')
decision_tests=[
    ('all_negative_and_below_threshold',[-1.]*15,.0012,-.001,True),
    ('one_score_tie',[-1.]*14+[0.],0.,-.001,False),
    ('one_score_worse',[-1.]*14+[.01],0.,-.001,False),
    ('p_exact_alpha',[-1.]*15,.00125,-.001,False),
    ('ci_exact_zero',[-1.]*15,0.,0.,False),
    ('ci_positive',[-1.]*15,0.,.001,False)]
for name,deltas,p,upper,expected in decision_tests:
    ns=dict(scores=pd.DataFrame({'delta_rmse':deltas}),
            boots={s:dict(p_worse=p,ci_family20=[-.1,upper]) for s in [7,101,2024]},ALPHA=.00125)
    exec(pass_code,ns);assert ns['passed'] is expected
    good('decision_'+name, expected=expected)
ns=dict(scores=pd.DataFrame({'delta_rmse':[-1.]*15}),
        boots={7:dict(p_worse=0.,ci_family20=[-.1,-.01]),
               101:dict(p_worse=.1,ci_family20=[-.1,.1]),
               2024:dict(p_worse=0.,ci_family20=[-.1,-.01])},ALPHA=.00125)
exec(pass_code,ns);assert ns['passed'] is False
good('decision_all_three_bootstrap_seeds_required')

# First-fold validator now checks actual expected day, closing v4 runner's narrower scope.
tr=pd.DataFrame(dict(farm=['F13','F47'],day=[3,5]));q=pd.DataFrame(dict(farm=['F13','F47'],day=[10,20]))
signature={'synthetic':True}
first=dict(status='PASS',signature=signature,raw_atol=1e-6,final_atol=2e-7,
           raw_errors={k:0. for k in ['repeat','single','reversed','other_query','fresh_fit']},
           scalar_maxdiff=0.,prefix=[],feature_causal=[])
for farm,td,qd in [('F13',3,10),('F47',5,20)]:
    for hour in [0,6,12]:
        first['prefix'].append(dict(farm=farm,day=qd,hour=hour,raw_maxdiff=0.,final_maxdiff=0.))
        first['feature_causal'].append(dict(farm=farm,day=td,hour=hour,feature_maxdiff=0.))
namespace['validate_first'](first,signature,tr,q);good('first_audit_valid')
for name,mutate in [
    ('wrong_query_day',lambda x:x['prefix'][0].update(day=999)),
    ('wrong_train_day',lambda x:x['feature_causal'][0].update(day=999)),
    ('nonfinite_error',lambda x:x['raw_errors'].update(repeat=float('nan'))),
    ('error_above_tol',lambda x:x['raw_errors'].update(repeat=1.1e-6)),
    ('wrong_status',lambda x:x.update(status='FAIL')),
    ('wrong_signature',lambda x:x.update(signature={'other':True})),
    ('duplicate_prefix_pair',lambda x:x['prefix'].__setitem__(1,copy.deepcopy(x['prefix'][0]))),
    ('wrong_final_atol',lambda x:x.update(final_atol=1e-6))]:
    bad=copy.deepcopy(first);mutate(bad)
    rejects('first_audit_'+name,lambda bad=bad:namespace['validate_first'](bad,signature,tr,q))

# Rebuilt manifest expression must match actual runner (only variable names differ).
run_preflight=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='preflight')
run_sig=next(n for n in ast.walk(run_preflight) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='sig' for t in n.targets))
verify_sig=next(n for n in ast.walk(functions['verify_complete']) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='sig' for t in n.targets))
normalized=copy.deepcopy(verify_sig)
for n in ast.walk(normalized):
    if isinstance(n,ast.Name):n.id={'runtime_core':'runtime','columns':'cols','dependencies':'deps'}.get(n.id,n.id)
for kw in normalized.value.keywords:
    if kw.arg=='baseline_sha256':
        kw.value=copy.deepcopy(next(x.value for x in run_sig.value.keywords if x.arg=='baseline_sha256'))
assert ast.dump(normalized,include_attributes=False)==ast.dump(run_sig,include_attributes=False)
good('all_manifest_fields_ast_match_runner_except_equivalent_baseline_path')
expected_run='136f72ce14cf7afc078185f0e1c421052910153d6f156ebe8eed604cc3a6d2f2'
assert hashlib.sha256((EXP/'run_v4.py').read_bytes()).hexdigest()==expected_run
good('pinned_run_source_sha')

result=dict(status='PASS_SYNTHETIC_AND_SOURCE_ONLY',verifier_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            run_sha256=expected_run, cases=records,case_count=len(records),
            numerical_comparisons=namespace['CHECKS'], actual_result_or_data_reads=0,
            actual_model_fit=0,actual_model_predict=0,actual_full_verification_runs=0,
            independent_bootstrap=actual_boot,
            limitations=['Verifier v1 original R3 replay omits saved train IDs, prediction digest, original labels, raw-component identity and original target bounds.',
                         'Source/manifest replay reuses feature/season implementation; it is not independent all-fold causal recomputation.',
                         'Stratified nonoverlapping sorted five-day blocks and finite 20000 draws inherit previous protocol assumptions.'])
with RESULT.open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
