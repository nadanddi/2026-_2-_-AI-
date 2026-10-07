# CH2 학습 배정 독립 검산 및 후속 범위 비평

검토일: 2026-10-07. 읽기 전용 코드·저장 receipt 검토와 현재 SHA 재확인만 수행했다. 모델 실행, 새 배정 실행, 정답 채점, GPU 사용은 하지 않았다.

## 판정

`ch2_training_assignment_crosscheck_v1.py`와 실제 `CH2_reference_independent_audit_v1.json`에서 기존 학습 전용 검산을 무효화할 핵심 오류를 발견하지 못했다. 이는 **고정된 CH2 비용식 아래 기존 학습 배정의 최적성 검산**이다. 물리적 원기록 연결, query 배정 정확도, EC 예측 개선의 PASS가 아니다. 다음 3개 후보는 아직 실행 허가용 규칙 등록이 완료된 것으로 보지 않는다.

## 실제 증거와 독립성

- receipt가 지목하는 7개 파일의 SHA256을 현재 파일과 PowerShell `Get-FileHash`로 다시 비교했다. 불일치 0이다. 모델이나 원 정답 값을 새로 계산하지 않았다.
- 저장 결과는 학습 5520행·230기록(F13 119/F47 111), 원 링크 201, component 46, cycle 17이다. 원 선택 링크 비용 차이 최대는 3.552713678800501e-15이다.
- F13 기존 배정과 독립 최적비용 차이는 0, F47은 1.1368683772161603e-13이다. Hungarian 구현이 SciPy를 호출하지 않고 augmenting path로 배정을 계산하며, 전체 2n×2n 행렬의 dual feasibility·선택 edge tightness·primal/dual 일치를 검사한다. 이는 선택 링크 몇 개의 비용 재현만 하는 검산보다 강하다.
- 2~6 크기 무작위 행렬 5개는 전수 순열과 비교하며, dummy 비용 검사는 4개다. 연결의 실질 대안 비용은 outgoing 12 + incoming 12 = 24다. 따라서 edge cost 13/20 연결이 허용되는 것은 코드 오류가 아니다. 이 목적함수에는 단독 edge<12 조건이 없다.
- 입력과 EC의 numeric 변환 전에 train ID 필터가 적용된다. query/gap 값 및 sub_temp의 numeric 변환 경로가 없다. CSV 파서는 비학습 행의 문자열을 읽고 hash는 파일 전체 바이트를 읽으므로, 정확한 표현은 ‘query/gap 값의 수치 해석·학습 사용 0’이다. ‘그 파일/문자열을 전혀 읽지 않음’으로 확대하면 안 된다.

## PASS의 경계

1. 원 선택 링크 201개의 비용을 독립 식과 비교했고, 원 배정의 독립 목적함수 값이 독립 optimum과 일치한다. 원 구현의 **미선택 모든 edge 비용**을 별도로 동일성 비교한 것은 아니다. 따라서 원 전체 cost matrix의 bitwise 동일성 주장까지 확장하지 않는다.
2. 비용식은 학습 EC 연속성을 직접 포함한다. 낮은 EC jump, 최적성, 작은 연결 비용은 같은 기준을 다시 평가한 결과이므로 원기록·진짜 날짜 복원 정확도의 독립 증거가 아니다.
3. graph cycle/component 검사는 저장된 원 successor 링크를 독립적으로 순회한 구조 검산이다. 새로운 최적 배정이 같은 링크를 선택하는지, 동률 최적해에서 graph가 유일한지 입증하지 않는다. 원 graph를 보존한 것은 타당하다.
4. FORBIDDEN=1e6은 유한 penalty다. 현재 dummy 대안 비용 24로 금지 edge가 이득을 얻지 못하는 구조지만, 일반적인 hard constraint로 표현하면 안 된다. 후속 API는 선택 edge가 금지 edge가 아님을 별도로 assert하면 좋다.
5. 한 BLK 학습 모집단에서의 비용 최적성은 원 TM/P2LOO/EL1의 학습 집합에서 같은 graph·scale을 재사용할 근거가 아니다. 각 fold의 허용 학습 입력/정답으로 재생성해야 한다.

## 다음 3개 후보 실행 전에 고정할 차단 항목

**CH2-N01 — 학습 topology와 query 배정의 목적함수 분리.** EC topology는 train-only 공개 정답으로 구성할 수 있다. query 선택 비용에는 EC, baseline 오차, 노출된 BLK 정답을 넣으면 안 된다. 원 CH2 `jump` 식을 query에 그대로 적용하는 API는 부적합하다. 입력-only 거리의 열·증분·결측 집계·정규화·tie 결정·margin/거부 기준을 숫자 결과 보기 전에 고정해야 한다. 새로운 문턱을 이번 BLK 결과로 고르면 새로운 탐색으로 기록한다.

**CH2-N02 — cycle/root·flank·숨은 시간의 의미.** cycle을 root 하나로 축약한 뒤 선형 시간 순서로 해석하지 않는다. cycle을 가진 전체 component를 baseline fallback하는 범위와 예외를 등록한다. left3/right3는 허용 train ID의 정렬된 고정 template이어야 하며, 뒤 query 입력을 training flank로 바꾸지 않는다. train EC graph가 같거나 가까운 것만으로 실제 하루/시간 거리나 제거한 gap의 빈칸 시각을 확정하지 않는다. 등록된 recordhour 가정으로 보간한다면 그 가정과 한계를 명시하고, 시간 순서가 결정되지 않은 경우 fallback한다.

**CH2-N03 — prefix 상태의 규정 경계.** 현재 기록 0..h와 같은 block의 이전 query 기록 중 이미 관측된 입력만 접근한다. h=0에서 현재 B1 또는 뒤 query h0는 사용할 수 없다. 현재 h0의 거리 계산에 h1이 필요한 식은 사용할 수 없으므로 결측/fallback을 고정한다. right training template의 0/1시는 허용 train 참조라는 별도 역할로 구분한다. query 상태에는 입력만 축적하며, EC 정답·선행 query의 정답·점수 잔차를 축적하지 않는다. 날씨 증분 scale은 train reference에서 한 번 fit하고 query 누적 상태에서 refit하지 않는다.

**CH2-N04 — 후속의 재현·다중비교 등록.** 고정 .2 endpoint blend, baseline/SG2 역할, shrink 적용 횟수, clip 순서와 fallback을 정확히 정한다. 최대 3개 방법이라는 상한 외에 RAW_PASS/QUERY_ROLE 및 seed 집계로 발생하는 실제 비교 개수를 등록하고 누적 탐색 수에 추가한다. 이번 진단과 기존 6개 narrow guard 실패를 구분하되 원 전체 검증기·모든 seed·최초 미사용 판정을 대체하지 않는다. 후속은 후보 선택 전에 source/등록/hash가 봉인되어야 한다.

## 필요한 API 감사

학습 graph/scale/cache 생성과 query transform을 분리하고, 학습 ID·graph digest·scale digest 및 각 query에서 소비한 입력 ID/시간/역할을 기록한다. 이후 규정 검사는 현재 prefix 결과를 단독·순서 역전·뒤 query/다른 farm/gap 교란과 비교해야 한다. 같은 block 앞 query 입력을 의도적으로 허용하는 상태형 방법은 단순 batch 독립성이 아니라 동일 관측 prefix를 재생한 결과의 일치로 검증한다. h0, missing increments, zero scale, 동률, anchor 부재, cycle component는 별도 합성 사례로 확인해야 한다. 이 감사와 실제 예측 검증은 이번 train-only PASS에 포함되지 않는다.

현 저장 검산 자체의 core blocker는 없다. 위 N01~N04는 아직 미실행 후속 후보의 실행 전 등록 조건이며, 지금의 산술 PASS로 닫히지 않는다. approval 서비스 credits 때문에 막힌 조립 실행을 다른 코드로 우회할 근거도 제공하지 않는다.
