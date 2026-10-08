# 원 PFN driver3 실제 등록 독립 검토

2026-10-07. 새driver/등록기/실제등록/sourceSHA만 검토했다. PFN/raw 모델·query·정답·성능·GPU 실행0, 부모worker 변경0이다.

**새 핵심 실행 blocker는 없다.** 실제등록231핀은 원PFN등록228핀을 모두 그대로 상속(누락/충돌0)+PFN등록2파일/driver3/등록기3이다. 현230개 기본읽기MATCH와 pandas1개 정식승인 SHA MATCH로231 전부 일치한다. producer2/driver3/원PFN등록2의 직접SHA도 일치한다.

producer2의 `predict_context`, `validate_saved`, cache/trace/rules/feature-loader 함수는 원모듈에서 import하여 그대로 호출한다. 함수 자체의 __file__는 producer2라 기존 audit codeSHA/registration2/contract/root_v2/2000reference×context5..8/CPU cached 모델정책을 유지한다. producer2.main은 호출되지 않는다. 새driver mainloop 변경은 as_posix output 집계1곳, 추가driver guard, runnerSHA 대상producer alias다.

등록기의 PureWindowsPath66fold×8files=528 합성은 oldstr의 slash prefix 실패와 as_posix의 fold8/전체528을 검사한다. model fits264/출력audit쌍528과 맞고 모델·receipt를 변경하지 않는다. 기존complete를 고쳐 덮어쓰지 않고 새fixedkey summary와 exactcompare하는 정책도 유지한다.

후속 독립 strict verifier/fullgate는 원PFN등록2뿐 아니라 driver등록3/currentdriverSHA도 transitive pin에 포함해야 한다. 비차단권고는 실제 importedproducer 기대HERE경로 명시, driver231 guard의 최종complete 직전 반복이다. 현producer predict_context마다 원228source 검사는 유지하지만 추가driver핀 검사는 시작시다. runtime resource/thread·메모리 운영은 source 적합성과 별도로 결정한다.

이 검토는 actual264 PFN fit/120trace/21numeric/528savedreceipt PASS가 아니다. raw/PFN 완료 뒤 fullmix/shrink한번/clip/RAW_PASS SG2·원행 전수 causal/독립 gate 전 정답parse 금지는 유지한다. 원24 전수·고정통계·최초미사용1회 판정도 남는다.
