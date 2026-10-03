# 제곱 목표로 분할한 숲의 원단위 평균 — 실행 전 고정

2026-10-03 집 코덱스. POWER2_ET DIAG10 시드7의 고EC 오차 감소·전체 악화 관측 후 작성한 별도 가설. 결과를 보고 정한 후속 실험이라는 출처를 숨기지 않는다. POWER2_ET의 설정이나 판정은 바꾸지 않는다.

RMS 복원은 mean(treepred²)의 제곱근 형태로, 원단위 평균보다 크다. 따라서 새 모델은 **분할은 EC² 목표로 학습하되 각 트리 예측을 sqrt(max(pred,0))로 원단위 복원한 뒤 평균**한다. leaf1에서 순수 잎의 트리 예측은 해당 학습 EC와 같아지고, 평균은 정상적인 원단위 조건부평균을 근사한다. 제곱 목표로 배운 분할만 남아 극단값에 따른 분할 품질 변화 여부를 확인한다. 동일 입력 충돌로 혼합된 잎에서는 완전한 동등성이 없으므로 일반 증명으로 주장하지 않는다.

- 단일 후보 POWER2_PARTITION_ET. 기존계절v2/22fold/3seed/모든15칸 유지. ET600/leaf1 기존동일, MASK FULL38/day대신동일season, query 현재이전 같은온실 입력만.
- 원단위복원 raw = mean_t sqrt(max(tree_t(X_imputed),0)). 모든 트리 동일 가중치.
- .48 ET구성원 교체: clip(ref+.48*(shrink(raw)-old_already_shrunk_ET),학습EC최소최대). 설정·비중·문턱 변경 없음.
- 기준 모든15칸 개선, DIAG온실층화5기록일bootstrap20,000회 p_worse<.025/3 및98.3333%CI상한<0. 지속목표family=SOFT_RESID10/POWER2_ET/이번안3개, 앞으로추가시family확장. 공개통과 전EL1미사용·이미소모final lock 금지. 최종채택에는EL1과새재현검사필수.
- 각행 sqrt(mean(tree)) ≥ mean(sqrt(tree)) Jensen 부등식 확인; 첫fold 원ET캐시재현·원단위트리복원 독립합·shrink/교체식 scalar검산·학습buffer/마스크범위감사.
- no new PFN fits/test predictions/submissions. ownlocal/ec_power_partition_20261003_v1 checkpoints only, same source hash.
