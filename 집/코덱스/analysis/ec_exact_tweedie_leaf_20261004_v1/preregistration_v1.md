# EC 정확 정규화 Tweedie 잎 · family24 실행 전 고정

2026-10-04 집 코덱스. 사용자 「1,2번 진행해봐」의 2번.

- 실제 EC 계절v2에서 LGB 한 구성원의 잎 값 계산만 바꾼다. 원 BASE14(remove day, append season), Tweedie rho1.5, lambda_l2=1, lambda_l1=0, num_leaves31, min_child40, 800trees, lr.03, subsample.8/freq1, colsample.8, seed7/101/2024를 유지한다. 트리 분할은 원 Newton gradient/Hessian 방식이다.
- 기존 .8R3+.2PFN, R3=.6ET+.3LGB+.1MLP, shrink.5, 원 학습 min/max clip을 한 번만 적용한다. ET/MLP/PFN 원 캐시·특징·문맥은 유지한다.
- 각 leaf의 실제 inbag 행과 직전 raw score F로 A=Σy exp(-F/2), B=Σexp(F/2)를 계산하고 `-A exp(-t/2)+B exp(t/2)+t=0`의 유일근을 구한다. λ1을 빼거나 fulltrain leaf·mean A/B로 바꾸지 않는다. 기존 C++ RenewTreeOutput hook을 통해 .03을 한 번 적용하고 실제 training score cache도 갱신한다.
- own-local 공식 LightGBM4.7.0 commit8f7036f03627054d5a54a6f965b13f4b9ff2cb63에서 원 Newton objective를 유지하고 별도 exact alias를 추가한 동일 DLL의 두 모드를 비교한다. 설치된 DLL/전역 라이브러리는 수정하지 않는다. Unicode 컴파일 경로 문제는 ownlocal 물리파일을 가리키는 TEMP ASCII 별명·subprocess GCC_EXEC_PREFIX로 해결한다.
- 원형 Newton 동일 DLL의 모든66셀 예측이 기존 원 LGB 공개 캐시와 absolute maxdiff≤1e-12여야 exact 단계로 진입한다. 실패 raw·원인·기존파일은 보존하고 tolerance를 완화하거나 부분 성공 셀만 골라 채점하지 않는다. 컴파일러 차이로 불통과하면 원 레시피 동등성 실패로 기록하며 정확잎 모델의 성능 기각과 구분한다.
- 최초 exact 실제 fit은 C++ inbag/leaf membership·직전 F·y(float32 label)·w·A/B·λ1root·.03 once·firstbias·800tree recursion/finalraw를 독립 Python으로 검산한다. fresh/repeat/order/single/other-query/원시특징 causal/prefix/scalar 감사 absolute1e-12 유지. A/B 집계의 relative1e-12는 합산 roundoff 용도이며 예측 문턱 완화가 아니다.
- 실제 sklearn wrapper가 훈련 Dataset을 해제하므로 첫 실제 fit의 내부 private score cache 직접 검산을 했다고 주장하지 않는다. 캐시 업데이트는 같은 DLL 합성 Newton/exact 두 mode 감사와 C++ 원 score UpdateScore 경로로 확인하며 실제첫fit의 F 재귀 추적으로 추가 확인한다.
- native model/checkpoint와 output/meta/SHA를 ownlocal 새파일로 저장한다. source/runtime/input/library/compiler/build/overlay/cache/manifest/preparation/preregistration 핀을 매 fit 전 확인한다. 실패/부분산출물 재시작 금지.
- 22fold/3seed=66candidate 완료·firstPASS·control66PASS·독립whole 검산 완료 후 15점수만 계산한다. 모든 시드×DIAG10/A/B/EXT10/EXT12 개선 + DIAG 각 seed20k farm별 5관측일 bootstrap p_worse<.025/24와조정CI [.025/24,1-.025/24]상한<0을 채택 조건으로 고정한다. 기존family20/21 규칙소급변경0.
- 이번과 FINAL_LOSS/RAW_LOSS 두안을합친신규3안 모두 누적family24 문턱을 쓴다. 공개검증 반복/새독립holdout아님. 전체 boosted ensemble의 정확한 전역 최적화라고 주장하지 않는다.
- 허용 데이터는 기존 공개 EC labels360일·허용 train_X이다. 원시EC train_y/잠금/EL1재점수/test예측·새제출 생성은0. 사용자 제출 지시 전 제출물 생성 금지.
