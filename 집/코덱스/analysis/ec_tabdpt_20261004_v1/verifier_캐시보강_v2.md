# 전체 verifier 원 R3 캐시 guard 보강 v2

2026-10-04, 집/코덱스. 비평 v18의 누락을 반영해 [verify_full_v2.py](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_20261004_v1/verify_full_v2.py)를 새로 작성했다. v1·run_v4·source·preregistration은 수정하지 않았다. 실제 후보 CSV/NPZ·감사·점수는 읽지 않았다. 기존 공개 R3 한 건의 JSON 메타와 기존 cache audit/engine 소스만 읽어 payload와 원 식을 확인했다.

## 추가 guard

전체 실행 시 각 기존 R3 cache를 다음처럼 확인한다.

1. 원 JSON의 prediction_sha256과 실제 NPZ SHA256을 대조한 뒤 배열을 로드한다.
2. kind/seed/fold와 원 provenance의 canonical key를 확인한다.
3. ordered train_row_id/query row_id, 유일성·학습/평가 분리, 각 원 멤버 배열 shape·유한성을 확인한다.
4. 저장 sub_ec를 fresh 공개 label map 및 query 목표와 대조한다.
5. **raw_r3 = .6 raw_et + .3 raw_lgb + .1 raw_mlp**를 np.array_equal로 정확히 확인한다.
6. 저장 train_row_id의 목표를 허용 공개 cache에서만 다시 매핑하고 fresh train 목표·원 metadata의 train_target_bounds·기존 baseline bounds를 대조한다. 학습/평가 행·일수도 확인한다.

66건 r3_cache_audit에 원 NPZ/JSON hash, 학습·평가 ordered ID hash, 공개 train 목표와 저장 query 목표 hash, bounds와 원 멤버 exact PASS를 기록한다. 원시 train_y나 잠금 정답은 읽지 않는다.

## 확인 결과

[verification_synthetic_v2.json](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_20261004_v1/verification_synthetic_v2.json): 합성 정상 cache가 통과했고 **학습순서·저장목표·멤버 식·파일 SHA·원학습 bounds**를 각각 오염시킨 5종을 모두 거부했다. --synthetic만 실행했으며 원 R3 NPZ나 후보 결과 파일을 실제로 로드하지 않았다.

[verify_v2_policy_static_v1.py](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_20261004_v1/verify_v2_policy_static_v1.py)로 v1/v2 AST를 대조했다. prefix_scalar/rmse_both/boot_diag/calculate_scores 및 seeds/folds/family/alpha/허용오차/source/runtime 고정 상수 모두 동일하다. 후보식·점수판정·bootstrap 문턱 변경은 0이다.

v1 SHA: e7891ee0dfcee43a24c12124b5075b05ee94c1775dbca7292a6f8f7323519eca  
v2 SHA: e5398b7c29eae32a4efc75381f7478cb1937e28bb4e61db1c1989016d732d9d9

비평가 /root/ec_harsh_review에 보강 내용과 합성/AST 결과를 전달했다. 부모 전체 실행은 v2의 --verify를 사용한다. 완료 뒤 기존의 full_verification_v1.json/full_scores_v1.csv/full_segments_v1.csv 이름으로 **새 파일만** 생성하며, 하나라도 이미 있으면 중단한다. 현재 이 하위 작업은 full 결과를 만들지 않았다.

## 범위의 한계

추가 guard는 저장 cache의 출처·행·공개 목표·원 멤버 산술·bounds 검산이다. 실제 66건 원 R3 cache 및 후보 결과 검산은 부모 실행에서 처음 수행한다. R3를 재학습하거나 원 ET/LGB/MLP 내부 학습을 재연하지 않는다. 저장 train_raw_r3는 shape/유한성을 확인하지만 train용 개별 멤버가 payload에 없어 학습측 3멤버 식을 재구성하지 않는다. 원 provenance의 전체 feature frame hash를 fresh feature bytes와 별도로 대조하는 검사까지 포함했다고 주장하지 않는다.

