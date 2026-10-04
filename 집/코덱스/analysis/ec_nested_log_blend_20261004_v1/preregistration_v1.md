# LOG 수정의 내부학습 혼합 · family19 · 2026-10-04 · 집 코덱스

학습 실행 전에 이 문서와 run.py를 main에 커밋한다. `--prepare`는 가용성/ID/cache 수식 감사만 하며 새 모델 적합을 하지 않는다.

## 동기와 범위

고정 LOG_PARTITION_MEAN은 기각됐으나 실제 계절v2와 저장 후보 사이의 혼합량0에서 공개 손실 미분이15/15칸 음수였다. 이는 가설을 만든 사후 진단이며 좋은 공개 혼합비를 선택한 것이 아니다. 기존 내부 보정·전문가 혼합과 유사한 틀의 후속이다. 원y leaf family18과 결합하지 않는다.

목표는 원단위 EC RMSE. 기존22분할/3시드7,101,2024/공개83160예측행을 유지한다. 실제 기준선 계절v2=.8R3+.2PFN, R3=.6ET+.3TweedieLGB+.1MLP, shrink/clip 순서 동일. 외부 후보의 LOG endpoint는 앞서 완료한 동일66캐시이며 source26c098b2d9cc0b18b816f37e0d8eb0985ae4d828bbcb945d0341ecf453c421f8로 고정한다.

## 단일 학습식

- matched CPU 작업의22CPU/88PFN/전체OOF/1584적합감사/첫CPU·PFN재현 PASS 후에만 새 적합을 시작한다. helper run_v2.py의 현재선택·원E·현재outer·ordered ID·중복/빈/누락·buffer·context·finite·정답·bounds 감사를 유지한다.
- outer 학습 fold 내부의 기존 한 holdout a/b만 쓴다. a의 MASK안전 FULL38(day 제외·season 마지막), 원순서, 시드로 원ET600과 LOG ET600을 새 적합한다. season 변환·21기록일 중심 이동중앙값 b는 **a만**으로 다시 만든다. LOG target=log(y/b), leaf ratio 산술평균, prediction=b_query×600tree 평균. h시 shrink는 같은 온실·같은 날0..h 예측만 사용한다.
- inner baseline `p0=clip(shrink(.8*matched_R3raw+.2*PFN4bag), a bounds)`.
- inner full_LOG `p1=clip(p0+.48*(shrink(LOGraw)-shrink(oldETraw)), a bounds)`.
- `e=p0-y_b`, `d=p1-p0`. d0을 포함한 **전체 innerquery 시간행**의 평균이다.
- `w=clip(-mean(e*d)/(mean(d²)+.01), 0, 1)`; 각 outer×seed 하나이며 두 온실/모든 시간에 공유한다. λ=.01,상한1,행 평균,게이트 없음은 결과를 본 뒤 변경하지 않는다.
- 이는 `mean((e+w*d)²)+.01*w²`의 제한 최적해다. 무벌점 RMSE 정확 최적해로 설명하지 않는다.
- outer prediction `clip(actual_v2+w*(stored_full_LOG-actual_v2), outer bounds)`.
- **이미 clip된 두 endpoint를 혼합**한다. raw수정량에 w를 먼저 곱하고 clip하거나 shrink를 다시 하는 다른 식으로 바꾸지 않는다. 같은 bounds의 bounded endpoint와 w∈[0,1]이므로 최종clip은 대수상 항등이어야 한다.

## 검증 및 중단 기준

- 직접 제공된 AGENTS 기준 유지: 모든15검증기×시드 칸 strict improvement + DIAG10 farm별5관측일block bootstrap20000의 P(worse)<.025/19, CI upper<0. w0/d0 동률을 개선으로 세지 않는다. k는 추가안 등록 시 더 엄격하게만 갱신한다. 판정 변경6.247 기록은 별도이며 이번 후보를 소급 채택하지 않는다.
- high≥1은 사후 진단만. 고EC31일/ordinary329일/후반46일/농가별/0시 손익과 편향을 함께 보고한다. EL1·새잠금·원시EC정답·test 예측·제출물 생성0. 통과해도 아직 채택/제출 후보로 확정하지 않는다.
- preflight 필수파일·정확한66캐시 예상ordered row IDs/labels/actualbaseline/scalar/bquery/hash 대응 검사 전 적합0. source/helper/support hash를 기록한다.
- 모든cell에 inner train/query IDs·labels·rawET/LOG/b/ratio/endpoints/e/d·mean numerator/denominator·gradient/KKT·w·bounds·clip빈도·농가/전후기 support·day/season span을 보존한다. inner train SSE/inner 혼합손실은 학습 진단이며 독립성능 증거가 아니다.
- 첫cell 동일ET 재학습·LOG tree재학습/leaf복원·배치·미래/다른농가 변조6개·수동prefix 확인. KKT·NumPy↔math.fsum·endpoint convex clip·scalar wholecell/finite/bounds 감사.
- 재개 시 완전한 CSV+NPZ+JSON+각hash+정확한metadata/input signature가 있을 때만 건너뛴다. 일부만 있으면 중단하여 새 버전으로 복구한다. 기존 결과를 덮어쓰지 않는다. 종료 시 신규/기존66감사를 모두 재취합한다.

## 알려진 한계

inner holdout 하나로 얻은 방향/비중은 outer와 분포·b anchor·span·방향크기가 다를 수 있다. 작은고EC 표본, 같은 공개검증 반복탐색, PFN문맥의시드공유, 고정λ의원EC단위 의존성을 유지한다. 내부 loss 감소는 학습특성이며 검증개선 증거가 아니다.
