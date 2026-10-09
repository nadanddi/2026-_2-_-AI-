# 최종 근거 문구 정정 v1
2026-10-09 · 이전 검토/독립검산 산출물 보존, 수치/알고리즘 변경 없음

1. final_independent_v5.json의 limitations에 남은 ‘day//5 is calendar blocks’ 및 이전 비평에서의 ‘달력구간’ 표현은 잘못됐다. row_id day는 각 온실의 상대 기록 번호이며 실제 달력 날짜는 제공되지 않는다. 실제 bootstrap은 farm×(day//5), 즉 상대 기록 번호의 5개 폭 구간이다. 이것이 연속 실제 달력5일임을 주장할 수 없다. PLAN_v3_reporting.md 및 문제설명서1쪽의 상대일차 정의가 정확한 근거다. 수치·draw·p는 그대로 유효하며 상대기록구간에 대한 bootstrap 진단으로 해석한다.

2. final_independent_v5.json의 ‘no proof of absent evaluation high days’는 ET 삭제실험 단독에 한정된 문구다. 별도 사전고정 집계 LB 하한 진단은 현재 공식 전행균등RMSE 문서 및 보관 제출CSV/점수쌍을 근거로 ‘모든 평가일의 일평균EC<=1’을 반박했다. 두 진단을 혼합하지 않는다. 평가에 고EC일은 존재하지만 ‘설명 안 되는 고EC일’ 존재/부재는 확인되지 않았다.

최종 보고는 이 정정과 critique_final_v1.md를 우선 적용한다. 원v5 및 기존비평은 수정하지 않았다.
