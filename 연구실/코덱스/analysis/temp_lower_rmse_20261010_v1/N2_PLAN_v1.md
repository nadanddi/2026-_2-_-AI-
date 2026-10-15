# N2 官方 타 농장 시간당 변화 prior · 적합성 검사 등록 전 계획

기존절대온도전이6.45/T-F1고정공기증분수축/6b.47대상증분손실/6b.48농장물리기울기와구별한다. 현재가설은절대수준을target으로학습하지않는것이며,현재절대입력도사용하므로수준정보가완전히없다고말하지않는다. 공식타농장Δ형태전이는target의추가이득을보장하지않는다.

## 고정된 적합성 검사
- 공식 train_X와train_y의row_id/sub_temp만읽고F13/F47(대상)/F32(기존근권가온반례)제외. 동일farm/day의h-1→h 연속시간쌍,두label 및현재/직전in_temp유한값이있을때만Δtarget을만든다. h0·다른날·기록불연속은제외.
- 최소96쌍있는농장만eligible. 전체eligible10개미만/총1000쌍미만이면지원부족으로진행안함. LOFO의분모는이조건으로fit전에고정한농장전체;농장별오차동률은실패.
- predictor=current14usable+동일날직전대비차분14+sin(hour*pi/12)/cos(hour*pi/12),총30열. farm/day/row_id/sub_temp lag미사용. Ridge100, medianimputer keep_empty_features=True(전부NaN열0유지),StandardScaler,train농장별동일총weight(평균1). 각LOFOfold의train으로만전처리fit. 결측은훈련median/empty0,임의backfill/미래보간0.
- 각농장LOFO MSE를zeroΔ와.5Δin_temp 둘다비교. 진행조건:eligible≥10/총쌍≥1000,농장80%이상에서둘다strictMSE개선,농장별동일가중macroMSE도두기준모두보다낮음. 동률/비유한은개선아님. cutoff/alpha/특징추가선택없이한번만검사. 이는출처간형태prior의적합성gate이며target온도채택검증아님.
- gatePASS일때만공식othereligible전체로priorfit. target의samefarm/day 현재·직전입력으로Δprior추론,day内h0=0으로누적한1열을CODEX잔차에추가하는후보를別途등록. current/previn_temp또는연속시간이없으면ΔpriorNaN,한날의누적途中欠損이면그시각부터그날끝까지NaN;日跨reset。MASK/day/task시각인과검사필수.
- target실험은원9W40G-S全混合paired검증으로연결한다. 구성원gatepass로성능개선주장/새제출하지않는다. baselineCACHE출처·fold/列/weights/season/sourceと実再fitprobeを検査. compatible아니면새full学習。温度DIAG10/EXT10/EXT12/새seed/복수비교 기준は別途fit前登録。

EC수치/대회외자료사용0,원데이터수정0. 지원부족/실패도해석갱신에반영한다.
