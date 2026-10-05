from pathlib import Path
import json,csv,collections,math
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
P=json.loads((H/'verification_v1.json').read_text(encoding='utf-8'));R=json.loads((H/'result_v1.json').read_text(encoding='utf-8'))
with (OUT/'case_registry_v1.csv').open(encoding='utf-8',newline='') as f:C=list(csv.DictReader(f))
segments=collections.defaultdict(collections.Counter)
for r in C:
    high=int(r['high'])
    for source in ['A','proxy']:
        z=segments[(r['v'],int(r['s']),int(r['hour']),source,r['farm'],int(int(r['day'])>=179))]
        sel=int(r[source+'_selected']);hh=int(r[source+'_hard_high']);miss=int(r[source+'_missed_high']);fp=int(r[source+'_hard_low'])
        z.update(n=1,high=high,tp=high*sel,fn=miss,fp=fp,hard_high=hh,captured_low_high=hh*sel,ordinary=1-high)
for z in R['segments']:
    got=segments[z['v'],z['s'],z['hour'],z['source'],z['farm'],z['phase']];assert all(z[k]==v for k,v in got.items())
extra=[]
for s in [7,101,2024]:
    g=[r for r in C if r['v']=='DIAG10' and int(r['s'])==s and int(r['hour'])==23]
    extra.append(dict(seed=s,hard_high_below_actual=sum(int(r['A_hard_high']) and float(r['prefix_A'])<float(r['y_day']) for r in g),hard_high_at_or_above_actual=sum(int(r['A_hard_high']) and float(r['prefix_A'])>=float(r['y_day']) for r in g)))
lines=['# 실제 일반 모델 A의 교차 예측 기반 경계 사례 정의','', '2026-10-06 집 코덱스. 이 결과는 기존 고EC 구별 실험에서 사용한 **seasonv2 A** 기준이다. 이후 SG2·DP1 조합 전체의 경계 정의라고 확대하지 않는다. 새 모델 학습이나 채택 결과가 아니다.','', '## 계산과 정의','', '저장된 실제 A의 원래 OOF 구성원을 다시 조립했다. R3(ExtraTrees 0.6 + LightGBM Tweedie 0.3 + MLP 0.1) 0.8 + TabPFN 4개 시드 평균 0.2, 이후 현재 raw 0.5 + 같은 날 현재까지 raw 누적 평균 0.5, 외부 학습 EC 범위로 자르기를 그대로 적용했다. 외부 검증 날짜와 ±1일의 학습 제외, PFN context가 외부 학습 행에 속하는지, 원래 구성원·입력·코드·환경 해시를 검사했다. 이번에는 원 모델을 새로 적합하지 않았으며 저장된 원 OOF를 재현했다.','', '각 시간 h의 점수는 후처리된 A 예측을 0시부터 h시까지 평균한 값이다. **23시 결과는 하루 전체 진단이며 오전 운영 성능으로 해석하면 안 된다.** 정답 하루 평균은 사례의 사후 분류에만 사용한다.','', '| 구분 | 고정된 정의 |','|---|---|','| 진짜 고EC | 정답 하루 평균 ≥ 1.0 |','| 예측 수준이 낮은 고EC 경계 사례 | 진짜 고EC이고 A prefix 점수 < 1.2 |','| 놓친 고EC | 진짜 고EC이고 A prefix 점수 < 0.9 |','| 포착했지만 예측 수준이 낮은 고EC | 진짜 고EC이고 0.9 ≤ A prefix 점수 < 1.2 |','| 일반날 오탐 | 정답 하루 평균 < 1.0이고 A prefix 점수 ≥ 0.9 |','', '**1.2 미만이라는 절대 수준의 경계 정의는 정답보다 낮게 예측했다는 잔차 정의와 다르다.** 아래 14일 중 정답보다 실제로 낮은 예측은 시드별 '+ '/'.join(str(x['hard_high_below_actual']) for x in extra)+'일이며, 나머지는 정답보다 높거나 같다. 문턱은 이전 실험과 동일하게 실행 전에 고정했고 이번 결과로 조정하지 않았다.','', '## DIAG10 하루 전체 결과','', '고유 360일(고EC 31일, 일반 329일), 시드 7·101·2024. 시드를 합쳐 1080개의 독립 날짜라고 주장하지 않는다.','', '| 시드 | 고EC 포착 / 31 | 놓친 고EC | 고EC 경계 / 31 | 포착한 고EC 경계 | 일반날 오탐 / 329 |','|---|---:|---:|---:|---:|---:|']
for z in P['summary']:
    if z['v']=='DIAG10' and z['source']=='A' and z['hour']==23:lines.append(f"| {z['s']} | {z['tp']} | {z['fn']} | {z['hard_high']} | {z['captured_low_high']} | {z['fp']} |")
lines+=['', '14개 고EC 경계 날짜는 세 시드에서 모두 동일하다. 일반날 오탐은 11개 날짜가 세 시드 공통이고 F47_117이 시드 7에서만 추가된다.','', '## 세 시드 공통 경계 사례','', '| 날짜 | 유형 | 정답 하루 평균 | 실제 A 점수 범위 |','|---|---|---:|---:|']
for z in P['DIAG23_cases']:
    label='놓친 고EC' if int(z['A_missed_votes'])==3 else '포착한 고EC 경계' if int(z['A_hard_high_votes'])==3 else '일반날 오탐' if int(z['A_hard_low_votes'])==3 else '일반날 오탐(1/3 시드)'
    lines.append(f"| {z['farm']}_{z['day']} | {label} | {float(z['y_day']):.6f} | {float(z['A_prefix_min']):.6f}–{float(z['A_prefix_max']):.6f} |")
lines+=['', '## 대리 모델과의 차이','', '| 시드 | 유형 | 실제 A | 대리 | 공통 | A에서만 | 대리에서만 |','|---|---|---:|---:|---:|---:|---:|']
for z in P['overlap']:
    if z['v']=='DIAG10' and z['hour']==23:lines.append(f"| {z['s']} | {'고EC 경계' if z['label']=='hard_high' else '일반날 오탐'} | {z['A']} | {z['proxy']} | {z['both']} | {z['A_only']} | {z['proxy_only']} |")
lines+=['', 'F13_241·243은 세 시드 모두 실제 A의 고EC 경계인데 대리 모델에서는 경계가 아니었다. F47_114·161 및 F13_133의 일반날 오탐도 대리 기준에서 빠졌다. 반대로 F13_125·137, F47_132·229는 대리 기준에서만 고EC 경계다. 특히 지난 분류기가 놓친 F13_137은 실제 A 점수가 1.2 이상이므로, 이번 정의에서 낮은 예측 경계로 학습할 사례가 아니다. 이는 정의 불일치의 근거이며 이전 성능 악화의 단독 원인을 증명하지는 않는다.','', '## 시간별·검증기별 확인','', '| 검증기 | 시간 | 날짜 수(출현 수) | 고EC 수 | 포착 범위(3시드) | 고EC 경계 범위 | 일반 오탐 범위 |','|---|---:|---:|---:|---:|---:|---:|']
for v in ['DIAG10','A','B']:
    for h in [0,6,12,23]:
        g=[z for z in P['summary'] if z['v']==v and z['source']=='A' and z['hour']==h]
        def span(k):return f"{min(z[k] for z in g)}–{max(z[k] for z in g)}"
        lines.append(f"| {v} | {h} | {g[0]['n']} | {g[0]['high']} | {span('tp')} | {span('hard_high')} | {span('fp')} |")
lines+=['', 'A·B는 검증 fold에서 같은 날짜가 반복될 수 있어 고유 날짜 수가 아니라 출현 수이다. 농장 × 1·2차 × 시간별 집계는 result_v1.json의 segments에 있으며 별도의 수동 카운터로 모두 재계산했다. 구간별 개선을 주장하는 실험은 아니다.','', '## 독립 검산과 반론','', f"63,864개 시간행, 10,644개 사례행, 3,548개 시드 합의행을 저장했다. 표준 라이브러리 CSV·math.fsum으로 혼합/누적 보정/범위 자르기, prefix, 하루 정답, 정수 TP/FN/FP·집합 교집합·시드 투표를 재계산했다. 실제 A 재구성 최대 오차 {P['maxdiff']['A']:.3g}, prefix 오차 {P['maxdiff']['prefix_A']:.3g}; 모두 1e-12 이내이며 원래 baseline과도 일치했다. pandas 설치 경로만 다르고 버전 및 라이브러리 초기화 해시는 동일했다(environment_path_audit_v1.json). 실행 진입점 이름 오류(run_v1)와 경로 비교 중단(run_v2)은 로그를 보존하고 새 버전에서 수정했다.",'', '| 주장 | 신뢰도 | 반론과 한계 |','|---|---|---|','| 실제 A와 동일한 OOF 기반 경계 정의 | 높음 | 저장된 구성원·문맥·해시·baseline 및 독립 산술 일치. 원래 입력 특징 생성 전부와 ET/MLP/PFN 학습을 이번에 다시 실행한 것은 아님 |','| 대리와 실제 A의 경계 사례가 다름 | 높음 | 동일 외부 날짜·시드에서 수동 집합 대조. 이전 실패 원인의 기여량은 미측정 |','| 이 정의로 새 분류기의 성능이 좋아짐 | 미검증 | 새 분류기나 고EC 전문가 결합을 학습하지 않았고 RMSE 개선을 측정하지 않음 |','| 30/31 포착이 새로운 데이터에서도 유지됨 | 미검증 | 반복 사용한 공개 DIAG10의 하루 전체 진단이며 새 홀드아웃 결과가 아님 |','', '## 다음 학습에서 지킬 조건','', '이 registry는 진단·사례 확인용이다. 전체 OOF를 그대로 다시 나눠 새 분류기를 검증하면, 분류기 검증 날짜의 정답이 다른 행의 원래 일반 모델 학습에 포함될 수 있다. 따라서 실제 학습에는 **각 외부 학습 집합 안에서 R3·TabPFN·후처리까지 동일하게 다시 교차 적합한 중첩 OOF**가 필요하다. 경계 사례 가중치와 분류 문턱은 새 실행 전에 고정해야 한다. 이번 결과로 문턱을 튜닝하지 않았다.','', '운영위 추가 안내는 저장소 요약에 기반하며 원 PDF 원문은 확보하지 못했다. 잠금 EL1·평가 정답·리더보드에서 계수를 계산하지 않았고, 새 제출 파일을 만들지 않았다.','', '## 실행 근거','', '- 정의: plan_v1.md / registration_v3.json / run_v3.py','- 독립 검산: verify_v1.py / verification_v1.json','- 표·세그먼트 재계산 및 반론: report_v1.py / report_checks_v1.json','- 실제 행·경계 registry·시드 합의: 집/코덱스/local/ec_actual_A_cases_20261006_v1/의 CSV 3개 (Drive 동기화 대상)']
with (H/'정의결과_v1.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
with (H/'report_checks_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS',segment_cells=len(segments),residual_definition_check=extra),f,ensure_ascii=False,indent=2)
print('PASS_REPORT',extra)
