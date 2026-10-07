"""Seal endpoint formulas and audit their information boundaries before scoring."""
from pathlib import Path
import hashlib
import json
import math
from blk_context_v1 import BLKContext, RAW, key
from blk_endpoint_methods_v1 import EndpointMethods

HERE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(name,obj):
    p=HERE/name
    assert not p.exists()
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
registration={
    'status':'SEALED_BEFORE_PERFORMANCE_PENDING_CRITIC',
    'layout_sha256':sha(HERE/'BLK_layout_v2.json'),
    'method_code_sha256':sha(HERE/'blk_endpoint_methods_v1.py'),
    'context_code_sha256':sha(HERE/'blk_context_v1.py'),
    'previous_trials':['6.345','6.346','6.347','6.348'],
    'difference_from_previous':'구조만으로 고정한 BLK에서 빈칸 양쪽 입력·양정답 제거 및 공개 양끝 정답을 함께 참조. 기존 단일 보류일 조건과 구분한다.',
    'methods':{
        'PAST_ENDPOINT':'0.8*baseline+0.2*왼쪽 공개 기록 23h EC',
        'BOTH_ENDPOINT':'0.8*baseline+0.2*양끝 EC의 원본 record-hour 선형보간',
        'CHAIN_PREFIX_GUARD':'동일 온실 train EC endpoint mutual best gap<=.05와 차순위 margin>=.005로 사슬 연결. 현재 query 0..h의 원14열 평균+현재값으로 train 사슬에 배정, 양끝과 같은 사슬일 때 BOTH, 아니면 baseline'},
    'fixed_unsearched_settings':{'blend_alpha':0.2,'chain_gap':0.05,'EC_second_margin':0.005,'input_coverage':0.75,'input_relative_second_margin':0.1},
    'settings_provenance':'연결 .05는 과거 단서 기준, .005는 과거 자정 차 단서의 척도. blend·배정 margin·coverage는 성능을 보기 전 임의 고정한 가설; 최적 계수라는 주장 없음. BLK 결과로 재조정하지 않는다.',
    'primary_baseline':'BLK_QUERY_ROLE',
    'diagnostic_baseline':'BLK_RAW_PASS',
    'baseline_not_yet_reproduced':True,
    'screened_variant_count':6,
    'seeds':[47,1414,6464],
    'normal_high_split':'사후 채점에만 공개 보류일 EC 평균 >=1을 high로 사용; 방법·문맥·배치에 전달 금지',
    'position':'0-based block index i: floor(3*i/n), 0=앞 1=가운데 2=뒤',
    'selection':'모든 seed×검증기 같은 개선 방향. BLK는8블록을 재표집 단위로 seed-mean squared error bootstrap 20000, seed2026100702, p_worse<.025/6. TM/P2LOO/EL1 유지. 새로운 seed/layout 최종1회는 아직 별도 봉인 전.',
    'limitations':['원 recordday 보간은 실제 날짜 보간과 다를 수 있음','EC 연속성 사슬이 진짜 동 정체성을 증명하지 않음','현재일 prefix만 사용하는 guard로 앞쪽 query 전체를 쓰는 더 강한 방법까지 시험한 것은 아님','BLK 모두 pass1; 실제 pass2 대표성 미확보','정답 사슬 수치 자체로 query 배정 정확성을 증명하지 않음']}
write_new('BLK_method_registration_v1.json',registration)
ctx=BLKContext(layout)
methods=EndpointMethods(layout,ctx.reference_inputs,ctx.reference_labels)
checks=0
for rid in sorted(ctx.query_ids):
    p=ctx.query_prefix(rid)
    result=methods.predict(rid,0.7,p)
    assert all(math.isfinite(result[c]) for c in ['PAST_ENDPOINT','BOTH_ENDPOINT','CHAIN_PREFIX_GUARD'])
    b,a=methods.blocks[key(rid)[:2]]
    left=ctx.reference_labels[a['left_23h']]['sub_ec']
    right=ctx.reference_labels[a['right_0h']]['sub_ec']
    assert abs(result['PAST_ENDPOINT']-(0.56+0.2*left))<1e-12
    t=result['calendar_weight']
    independently=(1-t)*left+t*right
    assert abs(result['BOTH_ENDPOINT']-(0.56+0.2*independently))<1e-12
    if not result['chain_guard_active']: assert result['CHAIN_PREFIX_GUARD']==0.7
    checks+=1
probes=[f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks'] for i in [0,len(b['query_days'])-1] for h in [0,12,23]]
for rid in probes:
    before=methods.predict(rid,0.7,ctx.query_prefix(rid));f,d,h=key(rid)
    future=[k for k in ctx._query if key(k)[0]!=f or key(k)[1:]>(d,h)]
    saved={k:ctx._query[k] for k in future}
    try:
        for k in future:ctx._query[k]={c:99999.0 for c in RAW}
        assert methods.predict(rid,0.7,ctx.query_prefix(rid))==before
    finally:ctx._query.update(saved)
    checks+=1
assert not(set(methods.inputs)&ctx.gap_ids or set(methods.labels)&ctx.query_ids)
write_new('BLK_endpoint_boundary_audit_v1.json',{
    'status':'PASS','checks':checks,'query_formula_checks':len(ctx.query_ids),'future_other_farm_probes':len(probes),
    'train_only_chain_links':len(methods.links),'train_only_chain_components':len({methods.root(d) for d in methods.days}),
    'heldout_truth_loaded':False,'baseline_used':'constant .7 FOR BOUNDARY TEST ONLY',
    'performance_evaluated':False,'full_baseline_causality_tested':False,
    'registration_sha256':sha(HERE/'BLK_method_registration_v1.json')})
print(json.dumps({'status':'PASS','checks':checks,'performance_evaluated':False},ensure_ascii=False),flush=True)
