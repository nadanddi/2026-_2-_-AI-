# ET 비교 계산 정정 — 최종 v3

독립 원 ET 재학습 검산은 raw 예측과 DI1 etS 캐시 사이 최대차 .1203106으로 실패했다. DI1 코드를 읽어 etS가 이미 p3.final(시간 평활화·clip) 출력임을 확인했다. 원 ET는 raw 상태에서 원래 tr 범위 안이므로 clip은 수치 변화가 없고 causal shrink 비교로 확인한다.

잘못된 초기 식 `baseline + shrink(.48*(new_raw - old_shrunk))`는 폐기한다. 의도한 ET 교체 식은 `baseline + .48*(shrink(new_raw) - old_shrunk)`이며 이후 전체 train 범위 clip을 유지한다. 두 안의 새 모델 학습·시드·입력·가중치·혼합비는 그대로다. replay_et_v3.py로 이미 저장한 new_et/old_et에서 후처리만 정정해 새 폴더 corrected_et_v3에 저장한다. run_v3.py는 처음부터 재현할 때의 수정 코드다.

초기 scores_v1/decisions_v1/result_v1 및 corrected_et_v2는 ET 두 안에 대해서 무효이며 최종으로 사용하지 않는다. 최종은 scores_v2/decisions_v2/result_v2, verification_v2.json, 실험결과_보고서_v2.md다. 모델 비교 10안과 family_k=10·판정 기준은 유지한다. 독립 검산에서 찾아낸 구현 오류의 정정이며 결과 기반 튜닝이 아니다. 앞서 사용자에게 전달한 EC_RARE_ET 중간 점수/편향도 이 정정으로 대체한다.
