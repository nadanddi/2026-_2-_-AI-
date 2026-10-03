# TabPFN-3.5 EC 구성원 교체 — 실행 전 고정

현재 모델은 TabPFN V2(로컬44MB checkpoint)이며, 설치된TabPFN9.0.0 코드는 V3_5를 지원한다. 공식 모델카드에 따르면2026-09-09 V3.5 기본checkpoint는 합성 데이터만으로 사전학습한 분류/회귀 공용 모델이다. 더 강한 사전학습 모델을 실제 EC 공개 검증으로 시험한다. 버전이 높다는 사실은 우리 데이터에서의 개선 증거가 아니다.

출처: https://huggingface.co/Prior-Labs/tabpfn_3_5 ; https://github.com/PriorLabs/TabPFN . 약관: https://huggingface.co/Prior-Labs/tabpfn_3_5/raw/main/LICENSE . 원문은 연구/평가 및 정의된 데이터 과학 대회를 비상업적 용도로 규정한다. 참가 대회의 모든 사용 조건을 법률적으로 확정한 판정은 아니다. 설치된model_loading._download_model은 V3_5 다운로드 전 ensure_license_accepted를 호출하고, browser_auth는 로그인·사용자의라이선스동의를 요구한다. 동의 대기 상태에서 승인 체크를 우회하거나 새모델을 다운로드/학습하지 않는다.

고정 모델: ModelVersion.V3_5 기본 `tabpfn-v3.5-20260909.safetensors`; Fast/다른checkpoint 비교 없음. 입력 season-last FULL38 동일. n_estimators4/seed1..4, **원래 V2 학습context_row_id 순서 그대로** 사용하여 표본 변경 없음. GPU float32/TF32off/highest, 불변성검사를위한고정2048 query배치+고정학습입력padding. 설치패키지/CPU기준선수정0. V2 장치 RNG어댑터는다른구조인신규V3.5에적용하지않음.

현재 season_v2의80% R3를기존rawcache로유지,20% PFNbag만 V3.5로교체: clip(shrink(.8*old_raw_r3+.2*new_raw_pfn_bag),outer_train_bounds). clip된season_pfn을다시평활화/역산하지않음. 원래4개raw_pfn과raw_r3로기준선을먼저정확히재현. 22fold/±1제외/360일/3R3seed/4PFNcontext고정. labels기존공개OOF만,원시EC/잠금/EL1/test값·예측/제출0.

판정family15:15개seed×validator전부개선,DIAG10 farm별5기록일20000bootstrap p_worse<.025/15와보정CI상한<0. 모든행의raw가중합/평활clip독립검산,첫fold새fit재현/8행prefix/순서/미래다른농장입력불변검사. 새GPU모델은CPU V2와수치동일한계산이라고부르지않음. 통과후독립후속확인전채택금지. 학습실행은weights파일과사용자라이선스동의가있어야함.

구체적인준비:prepare_baseline.py가현재공개22fold의V2원시캐시/표본/기준선을검산저장,run.py가이캐시를읽어신규PFN만학습한다. weights는내local/ec_tabpfn35_20261003_v1/weights/에만둔다. 모델로데이터를온라인전송하는API사용0,실행시HF/Transformers offline/telemetryoff.
