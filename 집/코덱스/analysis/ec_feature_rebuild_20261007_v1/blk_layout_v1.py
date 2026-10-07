"""BLK selection reads IDs/label availability only, never EC/temperature values."""
from pathlib import Path
import csv
import hashlib
import json
import random
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'공용/대회자료/정형데이터/참가자_배포'
SEED=2026100701
LENGTHS=[5,5,10,10]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ids(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:
        return [r['row_id'] for r in csv.DictReader(f)]
def key(rid):
    f,d,h=rid.split('_');return f,int(d),int(h)
xids=ids('train_X.csv');testids=ids('test_X.csv')
xhours={}
for rid in xids:
    f,d,h=key(rid)
    if f in ['F13','F47']:xhours.setdefault((f,d),set()).add(h)
labelhours={}
with (DATA/'train_y.csv').open(encoding='utf-8-sig',newline='') as handle:
    for r in csv.DictReader(handle):
        f,d,h=key(r['row_id'])
        # Availability only. Numeric label values are not parsed, ranked or scored.
        if f in ['F13','F47'] and r['sub_ec'].strip() and r['sub_temp'].strip():
            labelhours.setdefault((f,d),set()).add(h)
available={k for k,v in labelhours.items() if v==set(range(24)) and xhours.get(k)==set(range(24))}
lockpath=ROOT/'집/코덱스/analysis/codex_independent/ec_final_lock/locked_days.json'
locked={(r['farm'],int(r['day'])) for r in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
eligible_days=available-locked
pool={};selected=[];failed=[]
for farm in ['F13','F47']:
    for n in [5,10]:
        candidates=[]
        for start in sorted(d for f,d in eligible_days if f==farm):
            required={(farm,d) for d in range(start,start+n+8)}
            if not required<=eligible_days:continue
            q=list(range(start+4,start+n+4))
            candidates.append({'farm':farm,'length':n,'flank_left':list(range(start,start+3)),
                'gap_left':start+3,'query_days':q,'gap_right':start+n+4,
                'flank_right':list(range(start+n+5,start+n+8)),
                'support_days':list(range(start,start+n+8)),
                'pass':'pass1' if max(q)<179 else 'pass2' if min(q)>=179 else 'cross_pass'})
        pool[(farm,n)]=candidates
    used=set();rng=random.Random(SEED+(13 if farm=='F13' else 47))
    # Longer blocks first; choose a complete layout by deterministic backtracking.
    choices=[list(pool[(farm,n)]) for n in [10,10,5,5]]
    for candidates in choices:rng.shuffle(candidates)
    def search(i,chosen,occupied):
        if i==4:return chosen
        for block in choices[i]:
            span=set(block['support_days'])
            if span&occupied:continue
            answer=search(i+1,chosen+[block],occupied|span)
            if answer is not None:return answer
        return None
    pick=search(0,[],set())
    if pick is None:failed.append(farm)
    else:selected.extend(pick)

observed_test_blocks=[]
for farm in ['F13','F47']:
    days=sorted({d for f,d,h in map(key,testids) if f==farm});groups=[]
    for d in days:
        if not groups or d!=groups[-1][-1]+1:groups.append([d])
        else:groups[-1].append(d)
    assert sorted(map(len,groups))==sorted(LENGTHS)
    for group in groups:
        before,after=group[0]-1,group[-1]+1
        assert (farm,before) not in xhours and (farm,after) not in xhours
        assert (farm,before-1) in available and (farm,after+1) in available
        observed_test_blocks.append({'farm':farm,'query_days':group,'input_absent_gaps':[before,after],
            'EC_label_available_ends':[before-1,after+1]})

hidden=set();gaps=set();flanks=set()
for b in selected:
    f=b['farm'];hidden|={(f,d) for d in b['query_days']}
    gaps|={(f,b['gap_left']),(f,b['gap_right'])}
    flanks|={(f,d) for d in b['flank_left']+b['flank_right']}
assert not(hidden&gaps or hidden&flanks or gaps&flanks)
training=available-hidden-gaps-locked
assert flanks<=training
def rowids(days):return sorted(r for r in xids if key(r)[:2] in days)
result={'status':'REGISTERED' if not failed else 'STRUCTURE_INFEASIBLE','selection_seed':SEED,
    'selection_uses_numeric_labels':False,'test_input_values_used':False,
    'shape_per_farm':LENGTHS,'flank_min_records':3,'locked_records_excluded':True,
    'source_sha256':{n:sha(DATA/n) for n in ['train_X.csv','train_y.csv','test_X.csv']},
    'observed_actual_test_structure':observed_test_blocks,
    'candidate_count_by_pass':{f'{f}_{n}':{p:sum(b['pass']==p for b in blocks) for p in ['pass1','pass2','cross_pass']} for (f,n),blocks in pool.items()},
    'blocks':selected,'infeasible_farms':failed,'train_ids':rowids(training),
    'query_ids':rowids(hidden),'gap_ids_REMOVE_INPUT_AND_BOTH_LABELS':rowids(gaps),
    'endpoint_flank_ids':rowids(flanks),'hidden_both_label_ids':rowids(hidden|gaps|locked),
    'causal_query_contract':'same farm current record 0..h and earlier query records only; all fixed public training endpoints permitted; later query inputs forbidden',
    'constraints':['배치 고정 후 성능 때문에 변경 금지','gap rows는 raw loader부터 제거; 달력/사슬/검색/표준화에도 전달 금지',
        '정답 기반 사슬은 train_ids 내 EC만 사용; query/gap의 EC와 온도 모두 숨김',
        '뒤 endpoint 정답은 공개 학습 참조이며 뒤 query 입력과 구분',
        '사슬 신뢰도는 학습 내 진단, 평가 사슬배정 정확성/새정보의 증거로 순환 사용 금지',
        'TM/P2LOO/EL1 유지. BLK는 우선 진단 후보이며 주 검증기로 자동 교체하지 않음'],
    'previous_catalog':['6.345','6.346','6.348','6.350'],
    'difference_from_previous':'정답 양끝 고정·입력도 사라진 양측 gap1·동시8덩어리·뒤query접근금지의 새 고정 검증',
    'performance_tested':False,'baseline_reproduced_on_BLK':False,
    'limitations':['새 BLK 정답도 기존 관찰 정답이므로 독립 신규 자료 아님',
        '정확한 모양으로 만들 수 없는 2차 구간을 억지로 채우거나 결과를 보고 완화하지 않음']}
path=HERE/'BLK_layout_v1.json';assert not path.exists()
path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','candidate_count_by_pass','infeasible_farms']},ensure_ascii=False))
print(json.dumps({'selected_blocks':len(selected),'query_rows':len(result['query_ids']),'gap_rows':len(result['gap_ids_REMOVE_INPUT_AND_BOTH_LABELS']),'train_rows':len(result['train_ids'])}))
