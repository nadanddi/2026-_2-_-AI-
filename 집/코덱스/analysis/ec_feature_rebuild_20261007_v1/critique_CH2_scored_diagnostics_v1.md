# CH2 실제 BLK 진단 결과 독립 비평

2026-10-07. own 결과와 Decimal checker receipt만 읽고 관련 source/산출물 hash를 확인했다. 정답·query·모델·score·checker는 다시 실행하지 않았다.

## 판정

고정6variant는 모두 등록된 BLK screen FAIL이다. 가중·tie/선택 문턱·cycle fallback·alpha를 풀거나 유리한 구간만 고르는 사후 변경 없이 이번6진단을 종료하는 것이 타당하다. 이 실패는 ‘CH2 전체가 쓸모없음’이나 ‘graph가 잘못됨’을 입증하지 않는다.

현재 result/checker/gate/predictions/scorer2/score등록2의 SHA가 각각 저장 증거와 일치한다. checker receipt는1152RMSE·48blockSSE·6p/CI·정확6variant576cell key집합·6요약평균 PASS를 기록한다. 저장 전체cell의3seed delta를 별도 PowerShell로 평균해6요약값을 대조했고 차이0이다. 이 재확인은 결과 내부 산술과 hash 검사이며 원정답 성능 재실행이 아니다.

|scope|방법|평균 seed 전체 ΔRMSE|p_worse|3seed 방향|BLKscreen|
|---|---|---:|---:|---|---|
|QUERY_ROLE|FLANK_SOURCE_MATCH|-0.00119145098049|0.318684065797|모두 개선|FAIL|
|QUERY_ROLE|PAST_QUERY_PREFIX_STATE|-0.000280008123055|0.318684065797|모두 개선|FAIL|
|RAW_PASS|FLANK_SOURCE_MATCH|-0.00106063236117|0.318684065797|모두 개선|FAIL|
|RAW_PASS|PAST_QUERY_PREFIX_STATE|-0.000223016144465|0.318684065797|모두 개선|FAIL|
|두 scope 각각|CH2_REFONLY_GUARD|0|1|baseline 동일|FAIL|

등록 alpha는.025/60=0.000416666666667이다. ‘4안의 평균 RMSE 감소’는 관측 사실이지만 확증된 일반화 개선으로 보고하면 안 된다. guard의0은 효과가 없는 활성 방법을 시험한 결과가 아니라 적용0인 baseline identity다.

## 가장 중요한 한계: 순효과 한 블록

저장8block 평균seed loss sum을 확인하면 활성4variant 모두 배열 index3만 음수이고 나머지7개는 정확0이다. QUERY FLANK -0.316824149703, QUERY PAST -0.074828060117, RAW FLANK -0.298967923605, RAW PAST -0.0631333097966이다. 결과가 보여 주는 것은 **순손실 효과가 한 블록에 집중됨**이다. zero net sum만으로 다른 block의 모든 개별 prediction이 같다고 단정할 수는 없다. 모든 적용행이 한 block이라고 더 강하게 쓰려면 별도 저장 diagnostics/활성행 증거가 필요하다.

각 farm4block을4번 복원추출하므로 유일한 효과 block을 놓칠 확률은 `(3/4)^4=0.31640625`이다. 다른 farm은 net effect0이고 선택되면 음수, 놓치면 tie0가 된다. 따라서 네 variant의 p가 같은6374/20001=0.318684065797인 것은 손실 크기를 잘못 복제한 증거가 아니다. 동일 resample과 단일 negative block 패턴으로 설명된다. 개인95%와60보정 CI도 모두 upper0이다. 효과 크기가 커져도 coverage가 이 구조에 머무르면 이 block screen을 통과하기 어렵다. 이는 label 후 threshold를 완화할 이유가 아니라 독립 block 근거가 부족하다는 뜻이다.

세 seed는 같은 query·graph·block에 대한 baseline 차이이며 세 독립 farm/block 반복이 아니다. 두 scope도 같은 자료·적용 규칙을 사용하므로4번 개선이4개의 독립 성공을 뜻하지 않는다.

## 적용 범위·구간 설명

supported_rows는 GUARD0, FLANK25/1440(약1.74%), PAST14/1440(약0.97%)다. 이 수는 source 배정 정확도가 아니며 올바른 물리적 원기록을 찾은 비율도 아니다. PAST가 FLANK보다 보수적이라는 적용 범위 관측은 가능하지만, 모든 경우 더 좋은 방법이라는 비교 결론은 아니다.

QUERY scope의 일반1368행·고EC72행에서 FLANK/PAST 모두3seed RMSE 감소가 보인다. F13 및 앞/가운데에서 관측 감소, F47·뒤에서는 delta0이다. 고EC는3일뿐이고 구간들은 같은 한 효과 block을 다른 방식으로 나눈 것이므로 독립적인 반복 증거로 셀 수 없다. 고EC·앞부분·F13만 실행하는 정책을 지금 선택하면 사후 label/결과 선택이다.24hour 비교도 같은 기술통계 한계를 가진다.

## 다음 작업과 주장 경계

이번6안에 새 weight·margin·cycle 예외·endpoint/위치 조건을 덧붙여 같은 BLK에서 재시험하는 것을 권하지 않는다. 고정 실패·희소 coverage·관측 평균 감소·단일block 의존성을 모두 보존하고, 원래 남은 도메인24/원66fold/문헌·데이터 단서/최초 미사용 확인 계획을 진행한다. BLK 실패로 원래 의무인 원검증기를 생략하거나 전체 family를 제거하지 않는다. 기존 blocked domain action은 승인서비스 상태가 해결돼야 진행하며 이 진단으로 우회하지 않는다.

학습 EC 조건부 graph의 최적성, query 입력유사도 가설의 coverage, endpoint blend의 관측 성능은 서로 다른 증거다. 현재 결과는 narrow fixed6flank+cyclefallback+.2blend 구성의 screen failure다. 실제 물리적 chronology/사슬 복원 정확도나 모든 CH2 후속 가설을 반증하지 않는다. CPU cached/pass1 baseline의 제한도 유지한다. 바닐라 원자료 inventory 완결 및 전체goal 완료와도 별개이며 채택·제출·확정 구성은 만들 수 없다.
