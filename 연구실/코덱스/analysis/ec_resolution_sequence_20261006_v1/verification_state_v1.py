from pathlib import Path
from collections import defaultdict
import csv,json,math,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
assert (H/'최종혹독비평_v1.md').exists()
P=ROOT/'연구실/코덱스/local'/H.name/'stage34_rows_v2.csv'
with P.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
unique={r['row_id']:r for r in rows};assert len(rows)==10152 and len(unique)==3384
days=defaultdict(list)
for r in unique.values():days[r['farm'],int(r['day'])].append(float(r['sub_ec']))
support=defaultdict(int)
for (farm,day),y in days.items():
    assert len(y)==24
    support[farm,'pass1' if day<179 else 'pass2','高EC' if math.fsum(y)/24>=1 else '一般']+=1
matrix=[dict(farm=f,period=p,ec_group=g,days=support[f,p,g],status='観測済み' if support[f,p,g] else '未検証') for f in ['F13','F47'] for p in ['pass1','pass2'] for g in ['一般','高EC']]
for m in matrix:
    m['ec_group']='고EC' if m['ec_group']=='高EC' else '일반'
    m['status']='지원 있음' if m['days'] else '검증 지원 없음'
state=dict(execution='COMPLETE_PARTIAL_PILOT',numeric_audit='PASS',independent_final_review='COMPLETE',candidate='SCREEN_REJECT',candidate_reason='seed101 고EC 무악화 조건 위반',baseline='기존 실제 계절v2',current_EC14_efficacy='UNTESTED',pass2_protection='UNTESTED_ZERO_QUERY_ROWS',full_DIAG10_A_B='UNTESTED',full80_nested='UNAVAILABLE_IN_LAB',adoption=False,submission=False,unique_days=141,unique_rows=3384,seed_rows=10152,coverage=matrix,new_fit=0,source_rows_sha=hashlib.sha256(P.read_bytes()).hexdigest(),note='최종비평의 상태·분모 분리 권고를 완료자료에 적용했다. 다음 실험에서는 학습 전에 같은 coverage를 등록해야 한다.')
with (H/'검증상태_v1.json').open('x',encoding='utf-8') as f:json.dump(state,f,ensure_ascii=False,indent=2)
print('RECORDED_VERIFICATION_STATES_AND_COVERAGE',len(matrix))
