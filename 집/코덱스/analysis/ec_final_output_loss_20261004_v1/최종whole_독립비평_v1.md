# 실제 학습 전 독립 whole 검토

최종 정적 검토 수용. 대상은 `verify_full_v3.py` SHA `6875fc1d4062d52edbbda6733c72977d45fe9f52b3daf3046317042e212979bd`와 `verification_math_v4.py` SHA `6e16ecf93a3d15cf8fe0c77bda55f647c6a360668c694adbf461b0cc0a8152ba`이다. 이는 코드의 사전 검토이며 실제 후보의 검증 PASS 또는 채택을 뜻하지 않는다.

## 독립 실행 증거

`adversarial_whole_v1.py`와 `adversarial_whole_result_v1.json`: 정상 합성 체크포인트 replay 성공, all-NaN 열의 median=0 처리 성공, 미래행 변경에 따른 현재 prefix 불변 성공. 체크포인트 missing key/feature order/float32 weight/nonfinite weight/variance 변조/scale=0/학습수 변경, 반대·비유한 bounds, 중복 farm-day-hour, source signature 변경/seed 변경/미래 prefix day/비유한 fresh 오차/완화된 문턱/missing first key/false finite flag, duplicate JSON/NaN JSON 등 **19개 오염을 모두 거부**했다. 실제 데이터 읽기와 EC fit/predict/score는 모두 0이다. 합성 배열에 대한 NumPy forward만 수행했다. 원 source mismatch 사례는 first signature의 dependencies exact 비교를 실행한 것이며 실제 전체 source/runtime traversal을 재실행한 것으로 해석하면 안 된다.

## 수정 확인과 필수 gate

이전 all-NaN median finite assert 순서 문제는 math v4에서 해결됐다. 원시학습 BASE14 imputer/scaler moment 비교는 상대 1e-12이며 실제 예측과 산술의 절대 1e-12와 분리됐다. 모델 dtype/shape/finite 검사와 feature order, transformed feature hash를 함께 검사한다. 체크포인트는 신선한 네이티브 학습 재실행이 아니라 저장된 전처리·가중치의 독립 NumPy forward 재현이다.

whole은 132셀과 166320행 완료, fit meta/signature/checkpoint/CSV SHA, 각 모드83160 aggregate 및 combined 동일성을 확인한 후 외부 성능을 계산한다. receipt에는 runner/preparation/preregistration/whole/math SHA와 config canonical SHA가 필수다. original R3 cache guard의 caller provenance, PFN cache 및 입력·공개정답 결속을 유지한다. 초기 head=0 baseline, 학습 loss401개·last state, init SHA와 finite flags, first native 감사 기록 exact keys와 1e-12 문턱을 검사한다.

최종 RMSE는 fsum와 NumPy의 수치 및 개선방향을 비교하며, bootstrap은 동일 고정 draws를 독립 fsum 합산·수동 quantile로 확인하고 p와 adjusted upper CI의 부호를 확인한다. 고정 alpha=.025/24, 두 후보 각각 모든15 개선 및 seed별 DIAG gate를 유지한다. 원시 EC/test/EL1/새학습을 verifier에서 호출하는 경로는 없다.

## 남는 범위와 비평

- first 감사의 fresh/order/single/prefix는 저장된 기록의 schema/SHA/문턱을 확인한다. 새 fit 없이 native fresh 학습을 replay했다고 주장할 수 없다. 학습 로그의 모든 gradient finite 여부도 저장된 증언과 SHA에 의존한다.
- source와 runtime의 실제 전수 pin 확인은 preparation 및 실제 완료 뒤 whole 실행이 담당한다. 이 독립 합성 fixture는 실제132개의 완성 출력을 읽지 않았다.
- first prefix 내부 레코드는 필요 필드와 값·순서를 검사하지만 추가 임의 필드까지 금지하지 않는다. 해당 field는 채택이나 예측 계산에 쓰이지 않아 수용 차단 사유는 아니다.
- assertions 기반 검증이므로 실제 명령은 최적화 `-O` 없이 실행해야 한다. 현재 고정 실행 방법은 이에 부합한다.
- b만 훈련하는 단일 inner heldout MLP이며 전체 outertrain crossfit 잔차학습이라고 설명하면 안 된다. FINAL과 RAW를 비교하더라도 손실 선택의 인과 효과나 일반화 보장은 엄격한 사전 기준 통과 외에 주장하지 않는다.

root가 별도로 보고한 prepared 전수462cache/84392check 및 math synthetic20142 PASS는 root 실행 증거이며 이 보고서의 독립 실행 수와 합산하지 않는다. 실제 결과 채택 판정은 완료 whole 검증 이후에만 가능하다.
