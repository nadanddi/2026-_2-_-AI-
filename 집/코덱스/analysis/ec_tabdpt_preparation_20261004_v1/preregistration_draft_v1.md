# TabDPT .2 부품 교체 — 사전등록 초안 v1

작성: 2026-10-04, 집/코덱스. **DRAFT / family 미등록 / 실제 fit·predict 미실행.** family20은 담당자가 확정할 등록 제안이며 여기서 등록하거나 검증 결과를 열지 않았다. 이 문서는 완성된 실행 사전등록서가 아니다. 아래 미정 항목을 담당자가 새 버전에서 확정한 뒤 실행한다.

## 검증 질문과 단일 변화

현재 EC 계절 v2의 .8 R3(ET600/Tweedie LGB/MLP) + .2 TabPFN V2에서 **.2 PFN raw 예측만 TabDPT raw 예측으로 교체**한다. R3 학습법·가중치, 계절 입력, FULL38, train fold, 전처리의 시간/MASK 규칙, 하루 내 shrink, 마지막 clip은 동일하게 유지한다. 효과 크기·높은 EC 개선·전체 RMSE 개선은 아직 주장하지 않는다.

후보는 다음 한 식이다.

```text
candidate_raw = 0.8 * existing_R3_raw + 0.2 * TabDPT_mean_raw
candidate_final = clip(existing_shrink(candidate_raw, query_rows),
                       outer_train_y_min, outer_train_y_max)
baseline_final  = clip(existing_shrink(0.8 * existing_R3_raw
                                    + 0.2 * existing_PFN_V2_raw,
                                    query_rows),
                       outer_train_y_min, outer_train_y_max)
```

각 부품을 먼저 clip한 뒤 차이를 더하는 계산은 이 식과 다르다. 담당 실행 코드는 동일 배열의 **raw** 부품을 재사용하거나 기존 recipe 그대로 재현해야 한다. shrink·clip은 혼합 이후 각각 한 번이다. 기존 shrink는 같은 farm-day의 현재·과거 예측 누적평균과 현재 예측을 반씩 섞는다. 최종 한 행을 단독 계산할 때도 해당 시점까지의 동일 prefix를 보존한다.

## 고정 제안

| 항목 | 값 |
|---|---|
| 공식 코드 | TabDPT-inference v1.3.0 |
| 코드 commit | 97e5494431e9527c7edb31cb4dcfc5f00b232fdf |
| weight repo / revision | Layer6/TabDPT / a5ca6e01c0fa09ec68c73e958e5199d1932abb3a |
| weight 파일 / bytes | tabdpt1_3.safetensors / 252233296 |
| weight SHA256 | 97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb |
| estimator | TabDPTRegressor |
| constructor | normalizer=standard, missing_indicators=False, clip_sigma=8.0, feature_reduction=pca, context_reduction=retrieval, faiss_metric=l2 |
| 실행 | device=cpu, use_flash=False, compile=False, verbose=False |
| predict | output_type=mean, context_size=512, n_ensembles=8, batch_size=8 |
| 실험 seed | 7, 101, 2024 |
| 입력 | 기존 FULL에서 day만 season으로 교체한 FULL38, 기존 순서 |
| 최초 감사 | torch/FAISS thread=1, torch deterministic algorithms=True |
| 허용 오차 제안 | raw 절대차 ≤ 1e-6, 최종 절대차 ≤ 2e-7, 상대오차 사용 안 함 |

`clip_sigma=8.0`은 공식 constructor 기본 입력 처리이며 마지막 EC clip과 다른 단계다. API 이름은 `n_ensembles`이다. tagged [regressor.py](https://raw.githubusercontent.com/layer6ai-labs/TabDPT-inference/v1.3.0/src/tabdpt/regressor.py)에서 `mean` 경로는 각 앙상블 점예측을 평균한다. `full`의 logits 평균에서 mean을 재계산하는 대안은 이번 후보에 포함하지 않는다.

FULL38 순서는 [adapter_draft_v1.py](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/adapter_draft_v1.py)에 고정했다. 기존 core 상수와 AST로 독립 대조해 38개·순서 일치를 확인했다. weight의 feature limit는 아직 읽지 않았다. 최초 실행 시 limit가 38 미만이거나 학습 열 전체가 NaN이면 중단한다. 현재 초안은 PCA가 활성화되는 실행을 허용하지 않는다. train 행 수가 512 이하이면 retrieval 설정이 full-context 경로로 바뀌므로 중단한다. 문맥 크기를 결과에 맞춰 줄이지 않는다.

## 실행 전에 담당자가 확정할 항목

1. 현재 baseline과 동일한 A/B/DIAG10 등의 **정확한 fold manifest·행 순서·feature hash·검증기 구현 hash** 및 허용 공개 목표의 출처를 새 prereg 버전에 기록한다. 여기서는 데이터 파일을 읽지 않았다.
2. train 특징은 대회 MASK 규칙, 평가 특징은 같은 온실의 현재·이전 입력 규칙을 적용한다. 동일/동조 묶음과 날짜 의존성에 대한 기존 fold 제외 규칙을 유지한다. 외부 360일 입력에는 외부 정답이 없으므로 그 입력 자체의 RMSE를 보고하거나 채택 근거로 쓰지 않는다.
3. 후보 개수와 다중검정 family를 담당자가 등록한다. family=20이 실제 등록되면 DIAG10 기준은 p_worse < 0.025/20 = **0.00125**이다. 아직 family20으로 등록한 상태가 아니다.
4. 설치되는 분리 의존성의 정확한 version/파일 hash, 공식 소스 commit, 로컬 weight bytes/hash, 공개 접근·license 검토 결과를 기록한다. 현재 환경 정보는 [inventory_v1.json](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/inventory_v1.json), 미확인 공개 접근 항목은 [source_metadata_v1.json](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/source_metadata_v1.json)에 남겼다.
5. raw·최종 일치 문턱을 감사 실행 전에 담당자가 확정한다. 실행 후 문턱을 느슨하게 바꾸지 않는다. CPU에서 시작하며 GPU 전환·다른 context/ensemble 설정은 이 단일안에서 제외한다.

## 최초 일치 감사와 중단 조건

**점수 계산 전**, 고정 train/공개 query 배열로 다음을 검사한다.

- 같은 fitted object·같은 seed에서 두 번 raw 예측.
- 고정 첫 8개 query를 batch 8과 1행씩 계산한 raw 예측.
- 같은 8행의 순서를 뒤집고 원래 순서로 되돌린 raw 예측.
- 첫 query를 고정하고 다른 7개 query만 바꾼 첫 raw 예측.
- 새 estimator를 두 번 만들고 같은 train/seed로 fit한 raw 첫 8행 예측. 이 항목은 실행 담당자가 별도로 구현해야 한다.
- 최종식은 대상 farm-day의 동일 현재·과거 prefix를 유지하고 미래 행·다른 farm-day 행만 변경/삭제해 대상 시점 결과를 대조한다. 대상 시점만 떼어 shrink하는 비교는 사용하지 않는다.
- 전체 feature 생성의 현재·과거/MASK 인과성 감사, baseline raw 부품 재현을 통과한 뒤 점수 검증을 시작한다.

정상 shape·유한성, package/API/source pin, weight hash, 일치 문턱 중 하나라도 실패하면 중단한다. 실패를 score 개선으로 보정하지 않는다. [adapter_draft_v1.py](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/adapter_draft_v1.py)는 위 raw 첫 8행 중 반복·단독·순서·타 query 검사만 준비했다. fresh-fit 및 최종 prefix 감사, fold runner·통계 채택 판정은 아직 구현되지 않았다. 같은 object의 predict는 `self.X_test`를 바꾸므로 서로 다른 스레드에서 동시에 호출하지 않는다.

## 채택·보고 제안

기존 사용자 규칙에 따라 **모든 seed × 모든 기존 검증기에서 같은 개선 방향**과 family에 맞춘 DIAG10 p_worse를 동시에 만족해야 채택 후보가 된다. 전체 RMSE를 주 판단으로 유지하고 기존의 높은 EC 과소예측 등 오류 묶음은 고정 진단으로만 비교한다. 검증 결과를 보고 새 subgroup·threshold·weight를 선택하지 않는다. seed 평균만 좋아지고 일부 seed/검증기가 나빠지면 이번 단일안은 채택하지 않는다.

실행 시간·메모리·query 일치 결과를 점수와 함께 보고한다. 분산·동조/희귀값의 제한과 fold 수에 따른 불확실성을 함께 기록한다. EL1 재채점·잠금 정답·원시 train_y·test 예측·제출 파일·발표 자료를 이번 준비 단계에 만들지 않는다.

## 현재 확인 범위

[static_audit_result_v1.json](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/static_audit_result_v1.json)의 PASS는 문법, 비활성 실행 관문, FULL38 순서, 소스 hash의 정적 확인이다. **실제 모델의 반복/단독 일치 PASS가 아니다.** 신규 install·weight download·fit·predict·데이터 업로드는 모두 0이다. 카탈로그 6.48의 기존 TabICL 2.2.0 선별 탈락과 다른 모델이며, 과거 TabICL 실패를 TabDPT 성능의 근거로 쓰지 않는다.

