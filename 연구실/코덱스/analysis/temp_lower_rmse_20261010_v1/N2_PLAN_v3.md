# N2 공식 타 농장 시간당 변화 prior · 적합성 검사 등록 전 계획

기존절대온도전이6.45/T-F1고정공기증분수축/6b.47대상증분손실/6b.48농장물리기울기와구별한다. 현재가설은절대수준을target으로학습하지않는것이며,현재절대입력도사용하므로수준정보가완전히없다고말하지않는다. 공식타농장Δ형태전이는target의추가이득을보장하지않는다.

## 고정된 적합성 검사
- 공식 train_X와train_y의row_id/sub_temp만읽고F13/F47(대상)/F32(기존근권가온반례)제외. 동일farm/day의h-1→h 연속시간쌍,두label 및현재/직전in_temp유한값이있을때만Δtarget을만든다. h0·다른날·기록불연속은제외.
- 최소96쌍있는농장만eligible. 전체eligible10개미만/총1000쌍미만이면지원부족으로진행안함. LOFO의분모는이조건으로fit전에고정한농장전체;농장별오차동률은실패.
- predictor=current14usable+동일날직전대비차분14+sin(hour*pi/12)/cos(hour*pi/12),총30열. farm/day/row_id/sub_temp lag미사용. Ridge100, medianimputer keep_empty_features=True(전부NaN열0유지),StandardScaler,train농장별동일총weight(평균1). 각LOFOfold의train으로만전처리fit. 결측은훈련median/empty0,임의backfill/미래보간0.
- 각농장LOFO MSE를zeroΔ와.5Δin_temp 둘다비교. 진행조건:eligible≥10/총쌍≥1000,농장80%이상에서둘다strictMSE개선,농장별동일가중macroMSE도두기준모두보다낮음. 동률/비유한은개선아님. cutoff/alpha/특징추가선택없이한번만검사. 이는출처간형태prior의적합성gate이며target온도채택검증아님.
- gatePASS일때만공식othereligible전체로priorfit. target의같은 농장·기록일 현재·직전입력으로Δprior추론,당일h0=0으로누적한1열을CODEX잔차에추가하는후보를별도 등록. current/previn_temp또는연속시간이없으면ΔpriorNaN,한날의누적중간 결측이면그시각부터그날끝까지NaN;날 경계에서 재시작。MASK/day/task시각인과검사필수.
- target실험은원9회 W40G-S 전체 혼합 paired 검증으로연결한다. 구성원gatepass로성능개선주장/새제출하지않는다. 기준 캐시의 출처·분할·열·가중치·계절·소스와 실제 재학습 프로브를 검사. 호환되지 않으면 새 전체 학습。온도 DIAG10/EXT10/EXT12·새 시드·다중 비교 기준은 별도 실행 전 등록。

EC수치/대회외자료사용0,원데이터수정0. 지원부족/실패도해석갱신에반영한다.

## 연결 비평 반영
- 공식 타농장 일부 온도 정답의 정수해상도/1시간 평균 보간 관측(4.15/6.457)은 변화 target의 전이 한계다. 적합성 진단에서 해상도/0변화 비율/직선 연결 비율을 보고하되, 결과를 보고 좋은 농장만 다시 선정하지 않는다.
- LOFO에서 잘 맞아도 대상 F13/F47의 추가정보나 9회차 전체 성능 개선이 입증되는 것은 아니다. 원 입력 스키마·차분·누적 특징의 규정 검사와 전체 paired 검증을 분리한다.
## 코드 비평 반영
- 농장균등 weight는 Ridge 손실에만 적용한다. imputer/scaler는 fold훈련행 기준으로fit하므로 행수가큰농장이분포에더기여한다. 전처리까지농장균등이라주장하지않는다.
- 입력 inf는 NaN으로바꾸고fold train median/empty0로처리,예측/MSE유한값assert와JSON allow_nan=False를고정. LOFO train 농장키/행수 및원시OOF를남겨독립검산한다.
