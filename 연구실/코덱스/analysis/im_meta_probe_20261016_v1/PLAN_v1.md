# 결합 메타 원안 탐색 계획 v1 — 2026-10-16 연구실 코덱스

사용자 요청: IM 인계·나단9 점검의 제안1 실행. 기존 M_v2는 max 유지 + 전문가 추가였고 기준선 예산 초과로 중단. 여기서는 원안 교체형 메타의 타당성을 탐색한다. 기존 M의 기각 판정, 후속 채택 제한, R5 미개봉은 변경하지 않는다. 제출·채택용 후보가 아닌 진단이다.

- 입력: logit(p6), logit(ppre), rc, logit(E_f/E_ps2/E_g7/E_gr), 총7열. 기존 특징 캐시·이미지만 읽는다.
- 메타: StandardScaler + LogisticRegression(C=1, balanced, max_iter=3000), 판정 .5 고정. JPEG Q92 420 게이트 안만 교체, 밖은 p6 유지.
- 학습: h3/h3j/h3s/h3g/gptr train/gptr_src train. 전문가 학습에는 게이트 밖 흔들기 진짜 포함. 메타는 게이트 안+ok. H 전문가 h3/h3j만, E_gr 음성 h3 real/h3j/gptr_src, 양성 gptr train만. 모든 사진 파생본의 source와 group 폴드 일치 검사.
- 3분할 시드 1601/1602/1603, 각5폴드. GPT 학습 두 묶음은 다른 폴드. 각 전문가의 OOF 생성. 메타는 OOF 전체로 fit. 메타 자체 학습내 CV 점수는 내지 않는다(전역 OOF를 재분할하면 간접 누수).
- p6/ppre/rc는 기존 고정 모델 특징이며 완전한 group OOF가 아님. 고정 기반 모델 위 전문가 결합의 탐색만 가능. 원본DB 의존을 명시한다.
- 평가: R2/R3/R4, sealed real/brown/bright/qwen/v7, jig500, public21, R3 처리5종, gpt1/gpt3/sd2, gptr hold 및 짝 real hold, newreal/realg/realn. 전부 기존 노출자료, 새 봉인 확증 아님. 학습·평가 source/group 교집합 수 보고.
- 평가 때 전문가: 고정 배포 나단9 사용. OOF 대 배포 분포차는 진단으로 기록, 결과 보고 모델/문턱 전환 금지.
- 판정: 탐색 적합성만. 모든3seed에서 R2/R3/R4 균형정확도 >= baseline, 전체3개 중 최소1개 엄격개선; sealed 진짜 신규오탐합 <=2; jig500·각처리진짜 오탐 증가 <=1%p; pub21 전부정답; 외부 가짜 각 세트 새놓침<=1; gptr 짝 진짜 새오탐<=1. baseline 합집합 .5% 절대 상한은 적용하지 않음(그 기준으로 M_v2를 재판정하지 않음). 원본대조 제거 시뮬레이션은 실제 DB 제외가 아니므로 참고만.
- R2/R3/R4: 묶음 bootstrap 20000, 3seed 평균 개선차 CI 및 p(delta<=0) 보고. 반복 노출/다중비교 때문에 유의성 확증 주장 금지.
- 단계: 자료/분할감사 → 계획비평 → 첫seed 실행 → 중간비평 → 나머지seed → 최종비평/독립재계산/기록.
- 파일: 내 폴더 및 서버 ~/storage/nadan/codex/im_meta_probe_20261016_v1만 새로 생성. 타AI 캐시/모델/사진 변경 없음. R5 경로 읽기 없음. 새 이미지·GPU 작업 없음(기존 특징으로 CPU 로지스틱).
