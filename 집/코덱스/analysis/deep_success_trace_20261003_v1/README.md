# 성공·실패 심층 추적 산출물 안내

최종 해석은 `심층_추적보고서_v1.md`를 읽는다. 사용자 요청에 따른 사후 모델 진단이며 새 후보 채택·제출용 검증이 아니다.

- 최종 전체 모델 입력대체 영향: `whole_model_effect_summary_v2.csv`, `whole_model_effects_v2.csv`, `whole_model_effect_hours_v2.csv`. 온도 게이트까지 대체 입력에서 다시 계산한 버전이다.
- CPU 구성원·5개 대체일 민감도: `group_effect_summary_v2.csv`, `effects_v2.csv`. PFN과 게이트를 고정한 부분 모델 진단이다.
- 실내온도 이력 세분화: `temperature_time_details.csv`. EC 0시/이후 입력 세분화: `ec_time_details.csv`. 서로 다른 개입 결과를 합산하지 않는다.
- 실패 추적: `failure_interventions.csv`, `period_control.csv`, `rare_weight_diagnostic.csv`, `rare_weight_training_fit.csv`, `raw_temperature_shift.csv`, `alignment.csv`, `actual_feature_neighbors.csv`.
- 성공의 겹침·관계: `success_weather_overlaps.csv`, `event_temperature_phases.csv`, `event_relations.csv`, `relation_contrasts.csv`. 다중 사후 비교이며 보편적 규칙·인과 효과로 해석하지 않는다.
- 최종 검산: `verification.json`, `additional_verification.json`, `whole_verification_v2.json`, `pfn_supplement_verification_v2.json`. 재학습 원모델·시간별 합산·분할·원정답 대조와 별도 원소스 재현을 포함한다.

코드의 최종 전체 모델 수집기는 `collect_pfn_v3.py`, 검산기는 `verify_whole_v2.py`, 보고서 생성기는 `report.py`다. 앞선 버전은 이력을 보존한 것이며 최종 결과와 혼합하지 않는다. 특히 초기 EC 피처 순서 오류 산출물과 실패한 GPU EC PFN은 제외했다. `PERIOD_CONTROL_v2.md`가 날짜 제거 진단의 최종 안내다.

온도 PFN은 원 CUDA 모델과 축소 배치의 수치차가 최대 0.000179℃여서 입력대체 영향이 근사값이다. EC PFN은 원 CPU 실행 방식에서 8문맥 모두 원출력과 차0을 확인했다. 자세한 감사는 `PFN_BATCH_AUDIT.md`를 읽는다.

모델·문맥·대체 예측 캐시는 `집/코덱스/local/deep_success_trace_20261003_v1/`에 있으며 Drive 동기화 대상이다. 실행에는 공용 env 부트스트랩과 기존 공개 OOF/원모델 문맥 캐시가 필요하다. EC 잠금 정답은 읽지 않았다. 새 재현 실행은 기존 파일을 덮어쓰지 않도록 새 버전 디렉터리와 캐시 경로를 사용한다.
