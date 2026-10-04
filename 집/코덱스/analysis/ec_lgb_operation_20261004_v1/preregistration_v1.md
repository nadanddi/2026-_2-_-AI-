# family21 LGB_OPERATION_ONLY 사전등록 초안

원 actual 계절v2에서 Tweedie LGB 입력만 BASE day 제거·season 마지막 14열에 DP1 원9 운영특징을 붙인23열로 변경한다. 원 ET/MLP/PFN 재학습0. 원 core.lg 모든 매개변수(n_jobs4 포함)를 유지하고 threadpool_limits2. 시드7/101/2024, 기존22fold ±1일 purge, 공개 S.loadec만 사용한다.

후보 raw=.8*(.6원ET+.3새LGB+.1원MLP)+.2원PFN. 원 shrink .5 이후 train min/max clip을 한번 수행한다. clipped baseline에 delta를 더하지 않는다. 최초 원14열 LGB 캐시 재학습 차이≤1e-12, 새23열 repeat/fresh/order/single/other-query/prefix 및 원38열+운영9열 future·otherfarm 변경불변, scalar 식 차이≤1e-12를 통과해야 전체를 진행한다. 문턱완화 금지.

전체66셀 완료 전 점수0. 판정은5검증기×3시드 엄격15방향 개선과 각 DIAG10 시드의5일 farm block bootstrap 20000회(RNG20261003+seed), alpha=.025/21, p_worse<alpha 및 CI[alpha,1-alpha] 상한<0를 모두 요구한다. 이미 사용한 공개검증 반복 한계가 있으며 새 holdout 증거로 부르지 않는다. 제출/채택 파일은 만들지 않는다.

준비 manifest와 이 문서 및 source를 main에 등록한 뒤 root가 실제 실행한다. 부분 artifact/서명 불일치/기존 aggregate는 보존하고 중단한다.
