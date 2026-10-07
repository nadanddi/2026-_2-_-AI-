from pathlib import Path
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent
def write(name,obj):
    path=HERE/name;assert not path.exists()
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
with (HERE/'feature_candidates_v1.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
for r in rows:
    if r['stage']=='1':
        p=json.loads(r['parameters']);p.update(candidate_unit='전체 창/지연을 한 묶음으로 추가, 창별 우승자 선별 금지',
            operators=['current','lag','rate','second_difference','mean','std','sum','count','event_if_actuator'],
            target_member='ET only, other members fixed, integration audit required')
        r['parameters']=json.dumps(p,ensure_ascii=False)
        if r['family'].endswith('_flow'):r['previous_catalog']+=';6.185'
    if r['family']=='all_raw':r['previous_catalog']+=';6.224'
aux=[('M01','row_id_metadata','6.157;6.161','기록일/시각/온실 경로별 재학습·기준선 간접경로 분리'),
     ('M02','record_pair_prefix','C6.205;6.350','정답 사슬 없이 현재 prefix와 공개 학습의 쌍 구조; 계절 교란 통제'),
     ('M03','record_order_prefix','6.345;6.348','물리 연결을 단정하지 않는 기록 순서/간격의 정보 효과'),
     ('M04','causal_calendar','6.253;6.350','달력 fit은 학습만·query는 prefix만·정답기반 구조 금지'),
     ('U01','temperature_aux','C6.254;6.256','양정답 중첩 보류·전체 공개 온도자료·최근 지정 기준선으로 재검토'),
     ('U02','temperature_anchor','6.317;C6.254','현재 온도 대신 공개 학습 이웃 온도; PF1/PF2 결과 수령 후'),
     ('U03','temperature_crossfit','C6.254;6.256','outer/inner 양정답 보류와 전체 온실 이전을 분리'),
     ('U04','all_farm_input_aux','6.11;6.238','EC 정답 없는 49온실 입력의 fold 내부 보조 표현, EC 정답 합성 금지')]
for cid,fam,prev,diff in aux:
    r={k:'' for k in rows[0]};r.update(candidate_id=cid,stage='3',family=fam,source='대회 공개 원자료',previous_catalog=prev,
        difference_from_previous=diff,status='WAIT_STAGE2_AND_CAUSAL_AUDIT',parameters='{}',performance='UNTESTED');rows.append(r)
with (HERE/'feature_candidates_v2.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reg=json.loads((HERE/'preregistration_v1.json').read_text(encoding='utf-8'))
reg.update(initial_registry_count=len(rows),exploration_candidate_units='1단계는 family별 고정 묶음, 창별 성능 선택 금지',
    interaction_grammar={'pair_inputs':14*13//2,'operations':['product','ratio_a_over_b','ratio_b_over_a','lag_cross_product'],
        'lags':[1,2,3,4,6],'status':'다음 배치 등록 대기, 원열 함께투입 쌍91개와 다른 공간',
        'ratio_rule':'분모 0/비유한값은 NaN. 임의 epsilon이나 test 기반 스케일 금지'},
    CPU_baseline_equivalence={'raw_member_atol':1e-6,'final_atol':1e-6,'rtol':0,
        'tolerance_selection':'새 점수 전에 사전고정; PFN CPU/GPU 차이는 우승자 선택 전에 쌍 재계산',
        'failure_action':'변경 기준선으로 별도 이름 사전등록, 원 기준선 동일재현 주장 금지'},
    confirmation_independence='새 시드/배치는 새 정답이 아님. 기존 정답 반복 사용에 따른 선택 편향 잔존; 독립 미관측 holdout으로 부르지 않음',
    full_path_ablation='원열 현재값 제거와 달력/참조/서명/운영이력의 간접 경로 제거를 분리',
    documentation_status='실제 학습/참조/문맥/내부fold ID contract는 runner 실행 전 필수 생성. 현재 query ID만 보존됐으므로 fit 미허용')
write('preregistration_v2.json',reg)
fixes=[
    {'issue':'C01','fix':'도메인 candidate=한 고정 family 묶음. 창별 결과를 보고 우승자 선택 금지','evidence':'feature_candidates_v2.csv'},
    {'issue':'C02','fix':'원열 함께 투입과 곱/비율/교차지연을 다른 탐색 공간으로 명시','evidence':'preregistration_v2.json interaction_grammar','remaining':'실제 후속 후보 등록'},
    {'issue':'C03','fix':'row_id/쌍/순서/달력/온도3경로/전체온실 보조학습8개 등록','evidence':'feature_candidates_v2.csv M01..U04'},
    {'issue':'C04','fix':'실제 학습/참조/문맥/inner ID contract 없으면 fit 차단','remaining':'실제 runner ID 생성'},
    {'issue':'C05','fix':'원열 직접/간접 전체 경로 제거를 분리','remaining':'전체 처리 source audit'},
    {'issue':'C06','fix':'CPU 동등성 절대허용오차1e-6 사전고정, 실패 시 별도 기준선','remaining':'실제 재현'},
    {'issue':'C07','fix':'fresh seed/fold는 새 정답이 아님 명시, 탐색횟수 공개','remaining':'confirmation seed/layout 봉인'},
    {'issue':'C08','fix':'atomic checkpoint/strict schema/오염거부 selftest 구현','remaining':'실제 runner 및 process start audit'},
    {'issue':'C09','fix':'문헌 슬롯 출처 읽기·중복·차이 검토 후에만 활성화','remaining':'②에서 실제 문헌별 등록'},
]
write('critic_fixes_v2.json',fixes)
plan=(HERE/'PLAN_v1.md').read_text(encoding='utf-8')
plan+='''
## 독립 비평 반영 v2

critique_plan_v1.md C01~C09를 critic_fixes_v2.json에 연결했다. 해결된 문서 조건과 미완료 실행 조건을 구분한다.
도메인24개는 각 family의 모든 사전 창·지연·연산을 한 묶음으로 넣는 후보이다. 창별 최고 점수 선별은 새 후보 등록 없이는 금지한다.
원열91쌍 함께 투입과 곱·양방향 비율·교차 지연은 서로 다른 실험이다. 후자는 다음 배치로 등록하며 현재 실행되지 않았음을 표시한다.
온도·49온실·row_id·쌍·순서·달력8개 후보를 추가했다. 성능 효과는 미검증이다.
CPU 기준선 raw/최종 예측 동등성 atol=1e-6, rtol=0을 고정했다. 이를 못 맞추면 동일재현이라고 부르지 않는다.
새 시드·배치는 새 정답이 아니며 독립 미관측 확인으로 표현하지 않는다. 반복 정답 선택 편향은 최종 보고에 남긴다.
checkpoint 코드와 합성 오염거부 검사는 구현했으나 실제 runner·학습/참조/문맥/inner ID·전체 인과성 감사가 남았다.
원열 제거는 직접 특징만 제거한 경우와 달력·참조·운영 이력 등 모든 간접 경로 제거를 별도 시험한다.
'''
p=HERE/'PLAN_v2.md';assert not p.exists();p.write_text(plan,encoding='utf-8')
print(json.dumps({'candidates':len(rows),'stage3':sum(r['stage']=='3' for r in rows),'fit_allowed':False}))
