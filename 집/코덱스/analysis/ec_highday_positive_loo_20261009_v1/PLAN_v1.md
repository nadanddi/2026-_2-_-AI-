# 고EC일 한 날씩 제외하는 분류 진단 — 사전 계획

사용자제안: 고EC부분만 LOOCV. 학습은 일반+고EC 두class를 유지하고, 공개360일의고EC26일(일평균sub_ec>=1.2)을한날24행씩통째로holdout한다. 이모델은 고EC수치를예측하는회귀모델이아닌 기존독립binary classifier다. 목표/입력/모델/시드는카탈로그6.469와동일,검증묶음만바꾼다.

원자료공개360일8640행,예전ec_highday_classifier_20261009_v1의고정features/metadata/registration을sha로확인하고사용한다. 원train_y값추가load0,소비40정답·test값·이전대회자료사용0. 기존MASK적격RAW14+DP1+팬/차광/보온낮밤현재prefix73개. 같은농장현재일0..h입력만,다른농장입력/미래/정답은feature0. 행단위LOO가아닌일단위LOO다.

모든fold의공개360일train에서target농장day±1,다른농장day±3,잠금40일각같은농장±1을제외. 다른고EC도purge에따라빠질수있으므로항상25일고EC로학습한다고표현하지않는다. fold마다실제ordinary/high train일수를기록,두class유지assert. 전처리median 및모델은foldtrain만fit.

기존ET(300trees depth10 leaf24 maxfeatures.7,무가중class,일sampleweight1/24),같은3시드8383/1919/7171와같은Logistic(C1) 기준모델,prior해당fold고EC비율. 학습모델설정변경0/재균형0/threshold사후선택0. ET3seed평균과각seed를별도보고한다. 첫정렬고EC2일만실행→독립중간비평후남은24일실행,총26×4fit. 모델저장/최종400재fit/제출0(기존모델유지).

주h15,보조h0/8/23. 각일1score만보고해중복hours를독립일로합산하지않음. .5고정탐지수·재현율/FN·평균score 및고EC조건부Brier/양성logloss,분산/시드범위/농장×구간. .2는고정민감도만. 평가정답모두1이므로ROC-AUC/AP/precision/FPR/accuracy는제시하지않음(positiveAP1/precision1은무의미). 일반일성능은기존DIAG보조맥락이며새LOO의오탐률로연결금지.

같은26일기존DIAGhighscore를join하고새LOO탐지변화/score변화/양성loss변화를paired표로기록. foldtrain총일수·고EC일수동시에변하므로 '고EC자료더많이줘서좋아졌다' 인과단정금지. day단위탐지더됨은자료내진단만,채택판정없음. n26의농장층화5기록번호blockbootstrap(20000회,seed202610095) 양성Brier차이CI와p_worse는탐색범위/종전자료노출/새독립확증아님명시. 겹친train과상관된고EC일때문에독립표본가정한p를보강으로제시하지않음. 전seedh15 방향/3개기존검증회귀채택조건은이번positiveLOO로대체하지않는다.

질문해석: 결과좋아도현재입력만으로일부고EC패턴재현가능하다는진단. 결과나빠도입력정보완전없음/자료부족원인확정아님. LOOCV는한번에학습에사용하는일수를늘릴뿐새고EC자료를추가하지않는다. 특정시기연속사건일수는독립농업사건수가아니다.

고정코드/원자료/행키/분할sha와학습query별제외목록,trainclasscount,24queryrow,누수무교집합을저장한다. 계획→첫2일→전체26일 독립비평,최종별도산술탐지·Brier·paired·bootstrap및sourcehash검산후전달한다. 구분모델6.469/클로드MX1·MX2기각6.471와별도진단이며회귀gate성공으로확장하지않는다.
