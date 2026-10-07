from pathlib import Path
import json,hashlib
from blk_context_v1 import BLKContext,key
from blk_endpoint_methods_v2 import EndpointMethodsV2

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    p=HERE/name;assert not p.exists()
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
reg=json.loads((HERE/'BLK_method_registration_v1.json').read_text(encoding='utf-8'))
reg.update(status='SEALED_PRE_SCORE_V2',method_code_sha256={n:sha(HERE/n) for n in ['blk_endpoint_methods_v1.py','blk_endpoint_methods_v2.py','blk_sg2_refonly_v1.py','blk_baseline_data_v1.py']},
    previous_trials={'6.345':'자정 EC 연속성 단서','6.346':'학습 EC 연결','6.348':'앞날 찾기 기존 실패'},
    pipeline=['raw R3 .6ET+.3LGB+.1MLP, per seed47/1414/6464','PFN context5..8 CPU mean, FULL38','raw .6R3+.4PFN','one .5current+.5same-day-prefix shrink','train EC minmax clip','SG2 ref-only fit; scope QUERY_ROLE or RAW_PASS','train EC minmax clip baseline','endpoint blend on final baseline','train EC minmax clip candidate'],
    statistics={'loss':'per row mean across seeds of squared error; NOT squared error of mean predictions',
        'block_delta':'sum over rows of mean_seed(SE_candidate-SE_baseline), with row count',
        'bootstrap':'farm-stratified: sample4 blocks with replacement independently per farm, pool8 sampled SSE sums / pooled sampled row counts',
        'weighting':'row-weighted, so 10-record blocks contribute twice 5-record blocks when each sampled once',
        'draws':20000,'rng_seed':2026100702,'p_worse':'(1 + count(delta>=0))/(20000+1); ties count worse; MC estimate',
        'CI':'2.5/97.5 percentile of bootstrap mean squared-error delta','comparison_alpha':.025/6,
        'dependency_limit':'8 selected blocks; farm stratification does not establish independence of blocks sharing training references'},
    cross_validator_status='BLK diagnostic only until original TM/P2LOO/EL1 same-fold safe endpoint selector preregistered; full adoption forbidden now',
    graph_name='train-only EC endpoint similarity connected components, not proven directed chronology or greenhouse identity',
    guard_diagnostic_requirement=['component sizes','link count','same-component endpoint blocks','cycle components','guard coverage by hour/block','absolute nearest distances and finite dimensions'],
    critic_fixes={'M01':'baseline pipeline/ref-only SG2/scope/finalclip now specified, whole output audit still required','M02':'loss/bootstrap weighting/farm stratification exact; old validators pending hence BLK diagnostic only','M03':'undirected component terminology explicit','M04':'diagnostics required; no post-score threshold changes','M05':'immutable copies/structural assertions/independent raw ID weight recompute'})
write('BLK_method_registration_v2.json',reg)
layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
ctx=BLKContext(layout);m=EndpointMethodsV2(layout,ctx.reference_inputs,ctx.reference_labels)
checks=0
for rid in sorted(ctx.query_ids):
    f,d,h=key(rid);b,a=m.blocks[f,d];_,ld,lh=key(a['left_23h']);_,rd,rh=key(a['right_0h'])
    # Independent absolute hour arithmetic; do not reuse method's calendar_weight.
    numerator=(d-ld)*24+h-lh;denominator=(rd-ld)*24+rh-lh
    left=ctx.reference_labels[a['left_23h']]['sub_ec'];right=ctx.reference_labels[a['right_0h']]['sub_ec']
    expected=.8*.7+.2*(left+(right-left)*numerator/denominator)
    actual=m.predict(rid,.7,ctx.query_prefix(rid))['BOTH_ENDPOINT']
    assert abs(actual-max(m.lo,min(m.hi,expected)))<1e-12
    checks+=1
try:m.labels[next(iter(m.labels))]['sub_ec']=99
except TypeError:pass
else:raise AssertionError('mutable label reference')
write('BLK_endpoint_boundary_audit_v2.json',{'status':'PASS','independent_formula_checks':checks,'immutable_reference_check':True,
    'hidden_truth_loaded':False,'performance_evaluated':False,'full_pipeline_audit_complete':False,
    'registration_sha256':sha(HERE/'BLK_method_registration_v2.json')})
print('BLK endpoint v2 boundary/independent formula PASS',flush=True)
