# 실수 특징·PFN 문맥 전수 독립 검산 보완

2026-10-07. `critic_verify_pfn_provenance_v1.py`를 실행했고 exit0, `PASS_FULL_FEATURE_HASH_AND_40_PFN_CONTEXTS`를 확인했다. 새 fit0, 부분 성능 채점0이다. 기존 사전 보고서를 수정하지 않고 누락 범위를 이 파일로 보완한다.

10개 fold 전부에서 public OOF의 정답을 별도 stdlib csv reader로 읽었다. 같은 ID의 반복 라벨이 동일함과 8640개 라벨을 확인했다. 공개 ID에 해당하는 실제 입력 특징을 구성하고 준비 JSON의 train/query 순서대로 별도 재정렬했다. season.mapping을 직접 호출해 fold 학습 집합에 따른 season을 재현했다. train/query 각각의 FULL_R3 47열·BASE_R3 23열 실수 프레임 hash가 준비 서명과 전부 일치했다. PFN38열 hash는 원 R3 및 PFN 캐시 provenance와 전부 일치했다. 학습 clip bounds도 재계산해 일치했다.

40개 PFN 캐시 모두에서 다음을 확인했다.

- 준비서명·sidecar와 NPZ의 SHA 일치, fold/context seed 일치
- query 행 ID의 완전한 순서 일치, raw_pfn 출력의 길이와 유한값
- 해당 fold train 순서에 대한 seed1~4 RNG의 2000개 비복원 추출 index 재생 및 context_row_id의 전수 일치
- train/query38열 hash·행수 일치, n_estimators4/context_size2000/float32 inference/checkpoint SHA의 원 provenance 일치

독립 코드의 label reader·순서 구성·hash 계산·season 주입은 runner의 prepare/ordered/fhash/season wrapper를 호출하지 않는다. **immutable 현재 package의 features 함수와 원 season.mapping 알고리즘 자체는 재사용했다.** 따라서 새로운 특징 알고리즘의 독립 구현이나 PFN 재추론을 했다는 뜻은 아니다. 캐시 입력·문맥·예측 파일의 정합성 검산이며 정확도와 효과의 검증이 아니다.

`ABLATION_PROTOCOL_v1.md`도 확인했다. SELECTED_CASE_DEPENDENCE와 RESIDUAL_STATE_CONFUSION 모두 **후처리 전 raw ET24시간평균−실제24시간평균**으로 지정했고, 평활 ET는 보조 진단으로 남겼다. 이전 비평의 척도 모호성은 해소됐다. 사전 등록 커밋 자체는 부모 에이전트의 기록이며 이번 독립 검산에서 git 작업은 하지 않았다.

현재까지 사전 차단 결함은 없다. baseline360일 전수 결과, 실제출력을 통한 SG2 활성 경로/후처리/시드·ensemble 순서, 캐시 및 최종 채점의 독립 감사는 baseline 완료 후 진행할 범위로 남는다. 원 데이터·결과·코드와 이전 보고서는 변경하지 않았다.
