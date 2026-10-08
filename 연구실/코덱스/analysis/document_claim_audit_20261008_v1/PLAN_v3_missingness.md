# 결측 처리 보완 — 성능 결과 보기 전
실행 v3는 첫 CO2 유효성 검사에서 중단했다. F13_040은 h3..14 결측으로 유효차분12 기준 미달. 기준을 완화하지 않고 R1/R2 NA로 보존하며 known/unknown 분모를 함께 보고한다. unknown은 정상 집단에서 제외한다. K1/K2는 현재 실내온도 원관측도 유효해야 한다. 예측·정답 finite 및 BASE assert 추가. v2 인코딩 오류, v3 관측 불충분 중단 로그를 보존.
역할분류 열은 role_diagnostic_v1.py C의 10개로 고정, LogisticRegression random_state=20261008, pooled AUC와 fold별 AUC 모두 보고. 같은 묶음 양역할을 같은 fold에 넣고 train-only median/scaler 사용. 클래스/그룹 부족은 NA. 날짜간 실제 동 연속성은 판별하지 않는다.
