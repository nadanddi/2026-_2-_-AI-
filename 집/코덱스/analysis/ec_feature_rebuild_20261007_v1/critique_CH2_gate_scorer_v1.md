# CH2 gate·scorer 사전 독립 비평

2026-10-07. verifier2/scorer1/score등록1/Decimal checker1 source만 읽고 현재 SHA를 확인했다. 실제 query·model·gate·정답 채점·checker는 실행하지 않았다. runner 진행 중인 상태이며 실제 미래 gate/결과가 이미 PASS했다는 판정은 하지 않는다.

## 사전 봉인 및 truth 경계

score 등록의 scorer SHA·verifier SHA·inference등록3 SHA를 현재 파일과 새로 비교했고 모두 일치한다. 이전 비평에서 남긴 verifier의 실행 전 source 봉인은 이 별도 score 등록으로 보완됐다.32구간·누적60·alpha .025/60·20k/rngseed가 등록돼 있다.

scorer는 첫 stage 파일로 full gate를 읽고 status/허용 여부·25920scalar·1440prefix·360boundary 및 verifier SHA를 확인한다. gate 전이source/current등록source/pred SHA와 기존 baseline 일치·1440unique ID·train_y SHA를 확인한 **이후** query EC만 숫자로 바꾼다. inference source를 import하지만 import 시 main이 실행되지 않고 query/정답 loader 호출도 없다. source상 missing gate에서 truth parse에 도달할 수 없다. 부모가 보고한 실제 missing-gate negative exit1과 부합하되, 이번 독립 작업에서 그 테스트를 재실행하거나 개별 실패 receipt를 별도로 확인하지 않았다.

## 산술·통계 수용

6 variant(method3×scope2)에 각각3seed×32구간=96cell, 총576cell이며 baseline/candidate RMSE를 합하면1152수치다. 구간은 전체/일반/고EC/위치3/farm2/hour24다. 고EC daily mean>=1과 위치는 score 전용이고 inference에 전달되지 않는다. subgroup는 고정 기술통계이며 유리한 subgroup를 새 실행 정책으로 고르는 허가는 아니다.

primary loss는 각 행에서3seed의 squared-error delta를 평균한다. seed평균 예측의 squared error로 바뀌지 않았다.8block별 delta sum·행수로 집계하고 farm마다4block replacement를 뽑아 합계/총행수로 재표집한다. 짧고 긴 block이 있을 때 block mean을 무가중 평균하는 오류가 없다. plusone p/ties>=0, 양끝 percentile CI와 .025/60이 source/등록과 맞는다. all3seed 전체 RMSE 개선 조건도 명시됐다. 평균 RMSE 차이의 정렬과 bootstrap MSE loss는 목적이 다르지만 등록된 기술 순위/검정 분리이므로 계산 오류는 아니다.

32 기술 구간 각각을 새로운 확증검정으로 취급하는 것은 현재6variant/누적60 보정 범위 밖이다. 이미 노출된8block/pass1 자료, 작은 고EC 모집단과20k tail 정밀도 한계를 유지하고 BLK_screen_pass만으로 채택하지 않는다. 원validator·모든seed·미사용확정 판정은 남는다.

## verifier·독립 checker의 범위와 권고

verifier2의 전수 소비 ID/횟수·prefix SHA·diagnostic 재생·fallback bit equality·독립 dl/dr 산술 설계는 직전 비평대로 타당하다. source와 sample boundary360/fullprefix1440/scalar25920의 결합 검증이며, choose 거리식의 별도 독립 구현 검산이나 모든최종출력 미래교란 검사는 아니다. 실제 PASS 전 score를 실행하면 안 된다.

Decimal checker는 Decimal60 RMSE와 `(candidate-baseline)*(candidate+baseline-2*y)`의 factorized loss로 다른 계산 경로를 쓰며, random.choice 대신 명시 randrange로 동일 고정 재표집을 재구성한다.48blockSSE 및6 p/CI를 검산하는 설계는 의미 있다. 다만 아직 실행 전이며 다음 강화점을 권한다.

- 정확6개의 `(scope,method)` 고유 집합과 각96cell의 `(seed,segment)` 고유 집합을 강제한다. 현재 len6/len96만으로는 중복·누락을 모두 잡지 못한다. block sum 배열 길이8도 zip 전에 확인한다.
- mean_seed_delta_RMSE와 결과 전체 rows/blocks/high_rows/alpha/cumulative60를 독립 확인한다. 현재 checker는 cell/delta/p/CI 중심이며 요약 평균을 직접 대조하지 않는다.
- high/normal 분할의 Decimal daily 평균 계산은 localcontext60 밖에서 기본 precision으로 수행된다. Decimal60 주장 범위는 실제 RMSE/손실/boot 계산이다. fixed cohort의 경계가 달라지면 조용히 재정의하지 말고 등록된 float 판정과 독립 정확판정의 차이를 먼저 진단한다. 빈 구간이 있으면1152checks 강제는 실패하므로 고정 layout의 nonempty 범위를 분명히 한다.
- verifier와 scorer는 runner main의 Python3.12/base_context 실제 경로 assert를 호출하지 않는다. 현재 own file/source핀은 검증되지만 strictruntime 증거는 직접 확인 또는 runner runtime receipt 기록으로 보완하는 편이 명확하다.

위 항목은 현 scorer 산술을 무효화하는 core 오류의 발견이 아니라 gate·독립 crosscheck의 방어력/표현 범위 권고다. 실제 실행 증거가 없는 지금 발견된 새 core score-blocker는 없다. 향후 scorer는 actual full gate와 current SHA가 모두 통과해야 하며, 최종 독립 검산·결과 비평 전 성능 결론을 보고하면 안 된다. 도메인 조립 approval credits 차단과 별개이며 그 작업의 우회가 아니다.
