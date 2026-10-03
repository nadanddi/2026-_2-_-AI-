# EC ET 온실 표시 단일 특징 시험

2026-10-03 집 코덱스. 지속 개선 목표의 4번째 후보 FARM_ET.

코드 확인: 기존 core.FULL은 현재 입력·0시·누적·시각·계절을 사용하나 farm 표시를 포함하지 않는다. 과거 ec_high_day_mixture/ec_midnight_setpoint 등 다른 구조는 온실 표시를 사용했으므로 ‘온실 ID가 미시험 정보’라고 주장하지 않는다. 이번은 현재 계절v2에서 ET 구성원에 ID 한 열을 추가하는 단일변화 ablation이다. 단순 ID 추가가 손실을 줄일지는 미확인.

- 기준 현재 계절v2 저장 공개OOF. support.loadec 동일22fold×3seed(7/101/2024), DIAG10/A/B/EXT10/EXT12.
- 기존ET600/leaf1/max_features1.0, 기존 FULL38/day대신끝열season 이후 farm47=(farm=='F47') 1열 추가. EC 목표·결측대체·season·훈련일은 기존동일. 나머지구성원재학습없음.
- candidate=clip(ref+.48*(shrink(newETraw)-old_already_shrunk_ET),학습EC범위). 비중·문턱·설정 변경 없음.
- 전15칸 방향개선,DIAG 온실층화5기록일bootstrap20,000회 p_worse<.025/4 및98.75% CI상한<0. 지속목표family4(SOFT_RESID10/POWER2_ET/POWER2_PARTITION_ET/FARM_ET)이며 이전안도 통과 주장할 때 같은family로 재검산. 공개통과 전EL1 미채점, 소모final lock 금지. 최종채택에는별도EL1·재현감사필수.
- ID는 해당 query 자신의 제공row_id 온실부분만. 다른 온실의 query 입력/정답을 특징으로 사용하지 않는다. 전처리/학습은 허용fold 안에서만. MASK 현재이전 특징 유지.
- 기존ET 원캐시 재현은 앞선 c83ac91 첫 DIAG0seed7 동일cols/데이터/600trees 검사(maxdiff2.22e-16)를 참조하고 이번 바뀐 모델은 첫fold 독립재학습으로 확인.
- 예측·RMSE/fsum·고EC/일반/후반/온실별 오차·bootstrap·scalar shrink/교체식 검사. 실패 시 ID×기간 등 추가특징을 같은이름으로 사후 추가하지 않음.
- 새PFNfit/원시EC라벨/잠금/test예측/제출파일 없음. ownlocal/ec_farm_identity_20261003_v1 checkpoint,같은source해시만재사용.
