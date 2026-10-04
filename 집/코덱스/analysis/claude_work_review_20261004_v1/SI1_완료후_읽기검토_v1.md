# Claude SI1 완료 결과 갱신

2026-10-04 집 코덱스. 실험1 완료 후 최신 Claude commit d75189a와 ec3_SI1_sealed_active_interaction_v1.py/.log 및 10-04 작업일지·카탈로그6.297을 읽기 전용으로 확인했다. 다른 AI 코드 실행/변경0, 진행 중 모델 중복0.

SI1은 R3S에 밀폐 비율×난방/순환팬/(100−차광)의 고정 상호작용3개를 추가한 실험이다. 자체 절차의 기각을 유지한다. 공개 후기46일의 DIAG10w 시드평균 RMSE .32658795008894187→.3316440320239417, p_worse .8862를 독립 확인했다. 새 seed13/505/6060의 DIAG10w는 각각+.9319/+1.9345/+1.7969% 악화한다.

own review_si1_v2.py/result에서 공개2208행·6점수의 ID/공개8640행 label 정합, stdlib fsum↔NumPy RMSE와 작성자 pooled calendar day//5 bootstrap20k의 scalar↔vector p값을 검산했다. EL1은 문자열 단계에서 제외하여 재채점0, 원시 EC/test 읽기0, fit/predict0. 처음 CSV BOM 읽기 실패 v1은 보존하고 v2로 보완했다.

작성자 R3S는 actual 계절v2와 다르고, 같은46일의 두 배치는 독립 정답이 아니다. 작성자 pooled calendar block은 코덱스의 농장 층화·관측5일 block과 다르다. 따라서 이 수치를 코덱스 후보의 채택 검증으로 대체하지 않는다. 특정3상호작용의 실패를 제어 입력의 모든 정보가 소진됐다는 결론으로 확대하지 않는다. 신뢰도 높음: 공개 부분 수치와 기각 방향; 운영 기여나 정보 부재의 일반 주장에는 근거 부족.
