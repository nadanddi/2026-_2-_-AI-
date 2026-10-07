# 두 h0열 진단 구현·준비 사전 독립비평

2026-10-07 연구실 코덱스. `run_v1.py`/`core_v1.py`/`test_projection_v1.py`/`score_v1.py`/`audit_tree_v1.py`/`preparation_v1.json` 및 PLAN_ADDENDUM을 검토했다. 신규 fit0·부분 성능 채점0이다.

**현재 확정적인 사전 차단 결함은 없다.** main 사전 등록 후 계획대로30ET 진단을 진행할 수 있다. 이는 구현·준비 감사의 판정이며 bootstrap 결과·효용·채택 PASS가 아니다.

## 실제 입력·서명 독립 검산

`critic_verify_preparation_v1.py`를 실행했고 exit0, `PASS_TWO_H0_FIT0_PREPARATION_AND_MEDIAN_PROJECTION`을 확인했다. source·core·plan/addendum·원 준비/완료/receipt/runner·기준trace의 SHA 9개를 확인했다. 기준30개 캐시는 sidecar/prep/NPZ SHA·fold/seed·train/query순서·출력 길이/유한값과 일치했다.

10fold 모두에서 새45열 train/query 실수 frame hash를 별도로 계산해 준비 서명과 대조했다. 남은열 순서는 기존47열에서 정확히 `in_co2_h0`/`act_heating_h0`만 뺀 순서다. 동일 학습 입력의45열 median이47열 median의 해당 투영과 exact 일치했다. fold1/seed7은 기존 저장한 BASE imputer median과도1e-12 이내 일치했고, 실제 저장 float32 train/query X의45열 투영과 별도 재구성한 X/Q가 exact 일치했다. immutable features·season 본체는 재사용했으며 새 모델을 학습하지 않았다.

## 코드 검토

- `load_base`는 strict read-only다. 누락 시 fit 가능한 `B.cache_r3`를 호출하지 않는다. PFN은 B.prepare의 기존38열/hash/RNG 검증 경로를 사용한다. oldprep/캐시의 정합성은 기존 전수 감사에 연결된다.
- 새ET에만45열을 전달하고 같은labels/ID/seed/설정을 사용한다. LGB·MLP·PFN 출력, season, bounds, SG2 reference는 원기준에 고정한다. seed별 결과와 실제 raw 평균 ensemble을 별도로 후처리하며 BASE 원출력도 함께 보존한다.
- 변경 forest1개에 bootstrap=False/criterion=squared_error 및8행 batch exact 검사를 두었다. trace는 columns45와source/prep/SHA를 포함한다. 지원 평활은 정렬된 F47_161 단일24행 prefix이므로 적절하다.
- score는 전수 receipt/rows SHA와 lock 제거를 확인한 뒤 채점한다. 농장별 관측 day//5 블록을 동일 블록수로 복원 추출하며, 기준과 변경에 같은 draw를 적용한다. 전체 SSE/행수로 RMSE,20k/RNG32617·변화율/percentile/p_worse를 계산하고 block/draw/replicate 결과도 보존한다. DIAG screen은 전seed 전체감소 AND ensemble p_worse<.025, P2≥2%는 별도채택보류 flag다. addendum과 일치해 이전 정의 모호성이 해소됐다.
- audit_tree는 원 route helper를 사용해 새600tree의 전체train/query경로/count/weighted count/y평균/leaf/weight/예측을 재구성한다. 실제45열과 median투영·원cache·3level지원도 비교한다. root가 작성·실행하고 critic가코드/receipt를검토한다는 주체 구분을 유지한다.

## 사전 고정·후속 감사 보완

준비 JSON은 run/core/plan 중심으로 서명돼 있다. 아직 fit0인 현재에 score/audit_tree 및 imported leaf_core/routehelper SHA를 별도 구현manifest로 묶으면 재현 범위가 더 분명해진다. 이는 현재 계산 결함이나 실행 차단사항이 아니며 원prep를 바꾸지 않아도 된다. score 결과 receipt에는 score source SHA가 기록되지만 결과 전 등록과 같은 의미는 아니다.

현재 runner는 캐시 재개 기능이 없고 기존 NPZ가 있으면 assert로 멈춘다. 부분 실패가 발생하면 산출물/로그를 보존하고 새 버전에서 완료쌍만 source/prep/ID/SHA를 확인해 재사용한다. 완료 receipt의 고정69,120행/30fit 숫자는 독립 검사에서 실제coverage와 대조한다.

독립 결과 감사에서는30캐시/69120행, fixedothermembers, 실제ensemble/prefix/clip/SG2와候補·gate·delta, day/group/case/선택bias두판정,20k draw재생/블록sum/percentile/p_worse/screen/P2flag를 재계산한다. tree sidecar의arm/fold/seed/source·column/기준trace를 연결하고 root의경로감사receipt를검토한다. 실제ensemble과seed별 최종값평균 차이도 따로 확인한다.

프로토콜상 단일 arm k=1은 이번 사전고정 screen 안의 수다. 이전 공개결과를 보고 좁힌 가설이라는 사실이 없어지거나 전체 적응적 탐색이 한 번이 되는 것은 아니다. bootstrap20k를 독립 표본수로 세지 않는다. 두열공동효과·proxy정보잔존·feature차원변화에 따른 forest전역재구축과 물리인과 한계는 이전 계획 비평대로 유지한다.

사전 검토 완료. main 등록 전 source가 달라지면 새버전/준비서명으로 연결하고, 전수 완료 전 부분 성능을 계산하지 않는다.
