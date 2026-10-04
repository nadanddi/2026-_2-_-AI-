# GATE family25 완료 결과 독립 비평 v1

2026-10-05 집 코덱스. verification_GATE_v1.json, crosscheck_GATE_v1.json 및 새 learning verifier source를 읽었다. 이 비평은 완료된 산출물과 별도 Decimal/manual bootstrap 검산 receipt를 대조한 해석 비평이다. 본 에이전트가 원 CSV 전수를 다시 계산했다고 주장하지 않는다. 새 학습·native 예측·원시 EC/test/EL1/잠금 조회는 하지 않았다.

GATE는 REJECT가 맞다. DIAG10 RMSE는 seed7/101/2024 각각 3.314514%, 3.516610%, 3.078655% 악화했다. 독립 Decimal30 receipt의 RMSE와 whole 표시값이 일치한다. manual 5-observed-day farm별 bootstrap p_worse는 .98885/.98545/.98960이며 adjusted CI 상한은 모두 양수다. alpha=.025/26 조건에 크게 못 미친다. 15개 validator×seed 조합 중 엄격 개선은 0개다. A/B 모두 baseline과 정확히 같고 EXT12도 같다. 동일은 strict 개선이 아니므로 이들 결과만으로도 채택할 수 없다.

일반 329일은 각 seed -1.420175%/-1.709187%/-1.945554% 개선했으나 고 EC 31일은 +5.675116%/+6.140091%/+5.664030% 악화했다. pass2 46일은 +12.500185%/+13.348377%/+11.408263% 악화했고 두 농장 모두 악화했다. 이는 현재 gate가 적용된 범위에서 손해와 이익의 합계가 불리했다는 관측이다. 고 EC의 물리 기전, 특정 날짜가 원인, 입력에 정보가 없다는 증명으로 확대할 수 없다. 소수 수정행과 큰 손실의 가능성은 양 mode 완료 후 예정된 사례 진단에서 확인할 문제다.

v1 첫 실행은 DIAG fold7 eligible0에서 21셀 뒤 중단됐다. 실패와 partial을 보존하고 v2에서 빈 표본이면 baseline/delta0으로 사전 고정·재등록한 것은 정의되지 않은 supervised estimator를 안전하게 처리한 실행가능성 보완이다. 최초 실행과 완전히 동일한 prereg라고 표현하면 안 된다. 학습 비용 label은 이미 사용됐지만 외부 score는 0인 상태에서 고친 보완이다. empty fallback은 같은 farm/pass의 엄격 previous-day anchor 및 training reference에서 충분한 교정 학습행이 생기지 않을 수 있다는 설계 제약을 드러낸다. fallback과 pass 제한으로 변화가 없는 validator를 정보 부재 증거로 해석하면 안 된다.

이전 비평의 near-constant scaler 문제는 명시적 sklearn float64 bound로 수정됐고 learning receipt에 fit receipt SHA가 추가됐다. whole 최종 receipt에도 learning receipt SHA가 저장된다. saved coefficient replay 최대 오차 1.1102230246251565e-16, 66셀/83160행 완료, scaler/gradient 및 Decimal/manual bootstrap PASS는 산술과 실행 증거다. 모델 채택 PASS와 구별한다.

RIDGE26은 GATE 실패를 보고 새로 선택·튜닝한 대안이 아니라 prior c933 등록의 bounded continuous 대조다. 현재 RIDGE의 score는 읽지 않았다. 두 mode 모두 완료되고 고정된 strict15/alpha 문턱 검산이 끝난 뒤 최종 비교 비평을 새 버전으로 작성한다.
