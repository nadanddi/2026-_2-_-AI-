# EC v2 멤버 제거 4안 — 실행 전 사전고정

2026-10-02 집·코덱스. 새 정보원 X1~X3와 구별되는 **기존 앙상블 구성 진단**이다. 모델과 특징을 늘리지 않고 한 번에 멤버 하나를 제거한다. 코드/프로토콜 main 사전커밋 뒤 root가 실행을 허가할 때만 실행한다.

## 고정 기준선·4안

기존 v2 원시 예측의 가중치 ET .48 / LGB Tweedie .24 / MLP .08 / TabPFN .20. 각 안은 그 멤버 하나를 제거한 후 남은 기존 가중치를 합1로 재정규화한다. 가중치/특징/시드/채점행/후처리 튜닝은 하지 않는다.

| 안 | ET | LGB | MLP | PFN |
|---|---:|---:|---:|---:|
| drop_et | 0 | .24/.52 | .08/.52 | .20/.52 |
| drop_lgb | .48/.76 | 0 | .08/.76 | .20/.76 |
| drop_mlp | .48/.92 | .24/.92 | 0 | .20/.92 |
| drop_pfn | .48/.80 | .24/.80 | .08/.80 | 0 |

가설: 어느 멤버가 다른 멤버가 틀리는 행에서 오히려 같은 방향의 오차를 늘린다면 그 멤버의 제거가 모든 시드·검증기에서 좋아져야 한다. 반증: 특정 시드/검증기만 좋음, 제거 이득이 독립 날짜 재표집에서 불안정함. 기존 전체 v2보다 좋은 ET 단독/평균 비교나 멤버 불일치 구간 진단(카탈로그6.40·6.42 및 D5)과 달리 **4멤버 각각 제거**를 동일 레시피·22폴드·3시드에서 고정 검증한다.

## 데이터·검증 고정

- 기존 phase3와 동일 F13/F47 비잠금360일8640행. 잠금40일은 row_id에서 확인하여 sub_ec float 변환 전 제외. 학습에서는 잠금일±1 및 해당 검증일±1을 제거한다. 잠금 정답·점수0, sub_temp0, test_X 값/통계/가중치0.
- 공통 `ec_model_common_20261002_v1/common.py`의 prepare/split_fold. DIAG10 10폴드 + A5 + B5 + EXT10/12 각1, 총22. 동일 splits와 원래 인과 EWM EXT 정의를 보존한다.
- 시드7/101/2024. ET는 core.FULL38열, LGB/MLP는 core.BASE14열. 평가에서 NaN인5채널은 MASK로 제외한 기존 특징만 쓴다. 현재/이전 같은 온실 입력으로만 특징을 만든다. 결측대체/표준화는 fold 학습행에서만 fit한다.
- 모델 recipe 그대로: ET600trees/leaf1/max_features1.0, LGB Tweedie800trees/lr .03/31leaves/min_child40/목적분산1.5, MLP128,64/alpha .01/lr .001/max_iter800/early_stopping .12/n_iter_no_change25. 기존과 다른 것은 CPU n_jobs/threadpool2만이다. MLP 내부 early-stopping은 기존 학습행 내 규칙 그대로이며 외부 검증 정답을 쓰지 않는다.
- 후처리 순서는 **원시 멤버 혼합 → .5현재+.5당일현재까지 예측 평균(core.shrink) → 학습fold EC min/max clip**. 멤버별 clip을 하지 않는다.

## PFN 캐시 사용 전 사전조건

1. 매 fold·시드 ET/LGB/MLP를 동일 recipe로 재fit해 R3 raw=.6ET+.3LGB+.1MLP를 계산한다. final R3와 기존 finished R3 cache의 최대차≤1e-6이어야 한다. 불일치하면 즉시 BLOCKED, 현환경 baseline 재현 필요로 보고하고 **캐시 PFN 사용/후보 채점 중단**.
2. 공통 baseline이 cache_pfn_safe=True이고 기존 R3/V2 clip행수가 모두0이어야 한다. clip된 finishedR3/V2에서 PFN을 선형 역산하지 않는다. train 범위 경계에 정확히 붙은 행도 보수적으로 불능 취급한다.
3. clip없을 때 finished PFN=(finished V2−.8·finished R3)/.2. 공통 API의 pfn_finished와 이 독립 선형식 최대차≤1e-9를 확인한다. inverse_shrink(va,pfn_finished)로 원시 PFN을 복원하고 shrink roundtrip 최대차≤1e-9. 복원한 원래 v2의 cache 최대차≤1e-6.
4. 같은fold 다른3시드에서 finished PFN 최대차≤1e-9: 원 PFN bag은 R3시드와 무관해야 한다. PFN원시/finished값의 학습라벨 min/max 범위 밖 개수와 extrema를 모두 기록한다. 원래 PFN은 외삽할 수 있으므로 **새 멤버별 clip을 추가하지 않는다**. 후보의 가중치 변경 후 최종clip 빈도를 따로 보고한다.
5. 입력/분할/잠금/공통코드/기존core/실행코드/recipe/환경/각원캐시 해시를 감사한다. row_id 정렬이 동일한 캐시만 이용한다. 모든 행 finite, 인과 shrink를 검사한다.

## 판정·통계

- 같은 공통 규칙으로 비교하는 총5안=멤버제거4+별도 CatBoost1. 따라서 DIAG10 사전 p_worse<.025/5=**.005**, 각 시드99% bootstrap CI에서 RMSE차(후보−v2)의 상한<0. 효과 크기 컷 없음.
- 채택 선별조건: **3시드×5검증기15칸 모두 RMSE가 v2보다 낮음**, 위 DIAG10조건이3시드 모두 성립. 일부 검증기/시드만 좋으면 FAIL. 3시드 평균 점수도 기술통계로 별도 기록하지만 평균이 개별시드 실패를 대신하지 않는다.
- DIAG10 온실×기록일5일 block 재표집20,000회, seed918, 행수 가중 pooled RMSE. A/B 중복 날짜는 폴드 발생단위 점수와 고유행 평균예측 점수를 구별한다. 공통 evaluate 함수가 모든안 동일 기준을 적용한다.
- fold RMSE 평균/표준편차, 온실/기록후반(day≥179)/0~6시 등 사후 오차분해는 해석용이며 모델/안 선택에 쓰지 않는다. 기존 검증기 재사용 한계를 표시한다. 통과해도 제출/확정구성은 만들지 않는다.

## 실행·산출물·검산

- 22fold×3seed 전체 완료. 성공 조기중단/유리한fold 선별 없음. baseline/clip 복원 사전조건 실패에만 즉시 중단한다.
- 코드·프로토콜·로그·숫자 요약은 이 폴더, 재개용 예측 캐시는 `집/코덱스/local/ec_member_ablation_20261002_v1/`. 새 파일만 작성, 이전 파일 수정 없음.
- 완결 OOF, baseline/PFN복원/clip 감사, 공통 평가 csv/json, environment manifest를 남긴다. 제출CSV·test 예측·모델최종학습을 만들지 않는다.
- 최종 보고 전 analysis-verification: 원시비잠금라벨↔OOF 정합성, 독립 csv/math.fsum RMSE, 미리 고정한 분모/본페로니/모든시드검증기 판정, 캐시 복원산술을 독립 검산한다. 누수·과적합·재현성과 한계를 보고한다.
