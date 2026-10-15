# N2 첫 농장 LOFO 독립 중간 검토

2026-10-10. critic 별도무학습실제검산2회exit0. 공식train_X의row_id/in_temp, train_y의row_id/sub_temp만읽음. EC수치0/새fit0/외부자료0.

판정: **같은n2_lofo_v3로나머지47농장LOFO를계속가능**. source/runtime등록SHA와actual파일이같고F01cachebytesSHA=metadataSHA,30개Ridge계수/intercept와3686개prior가전부유한이다. rowIDs유일·F01prefix·predictshape·훈련271930/query3686·훈련47농장키에F01/F13/F47/F32없음을확인했다.

별도공식자료시간쌍재구성은farm/day/hour정렬후연속hourdiff1·h>0·finiteΔtarget/Δair·제외farm조건으로계산했다. eligible≥96인농장48/총275616쌍이root지원집계와같고F01의query3686 rowIDs도cache와전수순서일치했다. 표준math.fsum MSE는prior .19686445856659504 /zero .24932175800325557 /halfair1.0761984536082474로root첫농장결과와같다. 따라서첫농장의두간이기준strict개선은실제확인됐다.

first농장성과는전체80%분모·macro진행PASS가아니다. 48농장전체eligible를그대로고정하고나머지LOFO를마친후판정한다. 앞농장미달/개선결과에따른feature/지원분모/임계변경금지. 현재통과는타농장적합성진단계속승인이며targetfull후보등록/채택/제출판정과다르다.
