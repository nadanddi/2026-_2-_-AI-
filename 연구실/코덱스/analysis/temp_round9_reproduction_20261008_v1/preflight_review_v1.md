# 9회차 온도 재현 사전 독립 비평

2026-10-08 · 연구실 · 코덱스 critic. 모델 실행 0, 원본 변경 0. PLAN_v1, 원ZIP의 상위/온도 README, requirements, manifest_12, make_submission_v13_temp.py, temp_mask_v1.py, env.py/env_extra.py를 ZIP에서 직접 읽었다.

판정: **환경·출처 확인을 전제로 재현 실행 가능**. 사용자 직접 요청인 기존 제출 재현으로, 신규 온도 개선/채택/제출이 아니다. 공식 train_y만 학습하고 평가 정답을 읽지 않는 설계이다. 성능 점수나 채택 기준을 이번 작업에 새로 적용하면 안 된다.

## 반드시 보완할 증거

1. **동일 환경의 증명 필요**: README/requirements/manifest는 Python 3.12.4, numpy 2.5.3, pandas 3.0.1, scipy 1.18.1, sklearn 1.9.1, lightgbm 4.7.0, torch CPU 2.14.0, tabpfn 9.0.0을 명시한다. 실제 모든 버전·모듈 경로를 기록하고 불일치는 최종에 밝혀야 한다. 버전 차이가 있더라도 원소스 실행성 검증은 가능하나 정확 재현 원인 해석에 제약이 된다.
2. **프로젝트 import 탈출 점검**: 원소스는 `../analysis/codex_independent/2차`를 sys.path 앞에 넣는다. 보통 추출본 내 없는 폴더라 무해하지만 이 경로와 common/env/harness/resid_reset_features 실제 `__file__`을 기록하여 저장소 원본 모듈이 우연히 로드되지 않도록 확인한다. 코드 폴더 전체 bytes 해시를 실행 전후 확인한다.
3. **체크포인트는 동일 SHA만으로 충분하지 않음**: 사전 등록 SHA `2ab5a07d…`인 ZIP 공개 체크포인트를 캐시에 배치한 뒤 실제 TabPFN이 선택한 경로를 기록한다. V2 지정·CPU·4 estimator·문맥 2000·시드 1..8 원설정을 유지한다. 환경 wrapper에서 create_default_for_version, fit/predict, numpy RNG를 교체하면 원방식 재현이라고 할 수 없다.
4. **입력/고정표 해시**: 공식 CSV3개 SHA를 manifest와 확인하고 두 season CSV SHA도 확인한다. 소스가 AGRI_DATA를 읽으므로 명시적으로 추출본 data로 설정하고 기록한다. 읽기 대상 reference submission_04/06은 출력 EC 복사 및 변화 진단용이며 온도 학습에는 사용하지 않는다.
5. **중간/최종 분리**: BASE/CODEX 완료 로그는 학습 진행 증거이며 전체 재현 성공이 아니다. PFN8/출력 저장/전수 비교 종료 후 판단한다. 출력 EC는 submission_04이므로 현재 submission_14 EC와 비교하지 않는다.

## 판정 기준 비평

PLAN의 strict PASS = 1440개 sub_temp 값의 저장된 소수 6자리 전수 일치는 명확하다. 줄바꿈 때문에 파일 byte SHA가 다르더라도 숫자 전수 일치 여부는 별도 판정한다. 저장 전 npz pred와 6자리 기준CSV의 차이는 반올림 때문에 원래 존재할 수 있으므로 이를 실패 기준으로 혼동하지 않는다. 최종에는 기준CSV 대 재현CSV의 exact 차이 행 수/maxabs/RMS와 원시 npz 혼합식 산술 검산을 각각 보고한다. 사후 허용오차를 추가해 strict PASS로 바꾸지 않는다.

온도 README 제목의 submission_12 및 manifest_12 파일명은 내용의 name=temp_candidate_v13와 맞지 않는 오래된 표기이다. 실행 대상은 v13 소스이고 상위 README/최종 CSV SHA가 9회차와 연결되므로 현재 대상 식별을 막지 않으나 문서 정합성 결함으로 기록해야 한다.

temp_v13_checks 추가 학습을 생략하는 계획은 재현 범위에 타당하다. 이번에는 기존 규정 검사 재검증이 완료되었다고 주장하지 않는다. README의 검사 PASS 주장이나 결정론 주장은 문서상 주장으로만 다루고 이번 실행 결과로 따로 확인한다.
