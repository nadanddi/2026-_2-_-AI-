# 9회차 온도 재현 중간 독립 비평

2026-10-08 · 연구실 · 코덱스 critic. runtime_v1.json, preparation_v1.json, run_v1.py, compare_v1.py 및 task 모델캐시 파일 목록을 읽었다. 모델 중복 실행 0.

판정: **현재 실행 진행은 타당하며 아직 재현 성공 판단은 불가**. runtime receipt는 원소스 SHA e10c6e2d…/wrapper SHA/공식 입력 SHA를 준비 기록에 연결하고, env/common을 ZIP 추출본 온도/code에서 로드했다. 캐시에는 공개 `tabpfn-v2-regressor.ckpt` 한 파일만 있으며 2ab5a07d… 체크포인트 사전 해시가 등록되었다. 원소스와 모든 고정표는 실행 전 SHA 검사 대상이다.

실제 Python은 3.12.10이며 패키지 기록 Python 3.12.4와 다르다. numpy/pandas/scipy/sklearn/lightgbm/torch CPU/tabpfn 버전은 온도 requirements와 일치한다. Python 패치 버전 차이가 있다고 수치 불일치를 자동으로 설명할 수는 없다. 이후 전수 대조로 결과를 판단하고, 이 환경 차이를 최종에 남긴다.

원소스 `build_frames`는 학습 피처 생성 전 평가 입력을 NaN으로 만든 MASK 세계와 평가 피처용 실제 입력 세계를 구분한다. 고정 season 두 표는 SHA 검사 대상이다. 이는 소스 정적 설계 확인이며, 변조 입력 추가학습을 동반한 규정 검사 실행 완료를 뜻하지 않는다. 공식 train_y 학습은 기존 최종 제출 재현의 범위에 해당하며 평가 정답 채점 코드가 이번 wrapper/compare에는 없다.

## compare_v1.py 검토

- 1440개 ID 유일성·기준CSV/최종CSV 행 순서·유한값·Decimal 숫자 exact 차이 행 수·maxabs/RMS의 전수 비교는 적절하다. 숫자 정확일치와 SHA 바이트 일치를 구분하며 원시 npz pred 대비 기준CSV의 반올림 차이도 별도 항목이다.
- 게이트는 공식 test_X의 현재 in_temp로 독립 재구성한다. 평가정답이나 외부 점수를 사용하지 않는다. 출력의 EC는 원소스 submission_04 복사라 현재 제출 EC 재현이라고 주장하지 않는 scope 표기가 적절하다.
- 보완: npz의 `row_id`를 로드하여 CSV IDs와 전수 대조해야 배열과 행 대응이 증명된다. 현재는 shape와 혼합식만 확인한다.
- 보완: formula=np.array_equal은 같은 numpy 연산 재계산이다. 최종 독립 검산에는 표준 CSV/math.fsum 기반 PFN 8개 평균·행별 혼합식을 추가하고 수치 반올림 재확인한다. 이를 저장 전 부동소수점 bit 완전일치 필수로 오해하지 말고 오차 크기를 보고한다.
- `new_temperature_models_run`은 비교 코드가 독립 관측한 사실이 아니라 execution receipt/log를 전제하는 주장이다. 최종에서 PFN1~8 완료/BASE·CODEX 로그와 execution status를 함께 교차 확인한다.

현재는 build_frames 진행 단계이며 BASE/CODEX/PFN8 완료 증거가 없다. 이 단계에서 말할 수 있는 것은 입력·소스·런타임 연결 확인까지이다. 마지막 전체 소스 SHA/프로젝트 모듈 경로/체크포인트 SHA·출력 비교 종료 후 최종 판정한다.
