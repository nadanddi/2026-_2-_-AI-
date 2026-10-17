# N3 계획 독립 비평

2026-10-10. N3_PLAN_v1과N3_CONNECTION_REVIEW_v1를읽었다. criticfit0/EC수치0/외부자료0. 기존H1/H2의기각수치/코드는이번에독립재채점하지않고연결비평의직접검토범위로인용한다.

판정: **고정명세의구성원진단은타당하나코드의label격리·anchor계산을먼저검토해야한다.** 원두머리계열재발명/실패번복이라고소개하지않고현재달라지는y0target/Δ0physics/shape0차감을명시했다. 세변경의각기여는실험한번에서분리되지않는다. 사전5후보alpha.005/중앙99%CI/3검증기방향/full단계별도등록계약은명확하다.

## 코드 전 확인

- 모든key는farm+rawt//24로만묶고season으로변한day를groupby하지않는다. train/query각day24행·hours0..23유일·h0유일assert. target생성은tr에서만y0table을만들고tr.rowIDs/key에일치시킨다. query에서sub_temp및유사label열을제거한frame만predict함수에주고querylabel을NaN/임의변조해도예측동일한negative검사를한다. h0truth는모든predict종료후진단함수안에서만사용한다.
- level은trainh0 1행당PHYS19+season20열Ridge와CF89잔차LGB이고w0만. levelquery는queryh0입력prefix만쓴다. rowweight숫자/leveltrainqueryday수를receipt에남겨기존daymeanweight나검증정답levelmodel로오인하지않는다. 미관측관리상태의h0추정지원이하루한행으로줄어드는제약을유지한다.
- shapephysics는currentPHYS열−같은rawdayh0PHYS열이다. 원NaN전처리보다차분이먼저, h0는정의상0. medianimputer의keep_empty_features/empty0규약도고정하고scalerwith_meanFalse/RidgeinterceptFalse를assert한다. 실제0벡터physics예측0·scale0벡터불변검사로의도와구현을연결한다. shapeCF89는원prefix·season규약을그대로사용한다.
- shape_raw0는같은querydayh0의**모델예측**이며정답이아니다. h0rawshape를0으로강제하는지최종에서빼는지다르므로현재명세대로raw0를빼최종shapeh0=0/최종predh0=level_hat를exact검사한다. rawshape학습target과차감후실제목적불일치를결과후계수로고치지않는다.
- 원baseline캐시SHA/row/order/keys/weights/season/source를연결하고firstbaseline재fitexact. 신규N3머리구조는4개estimator계열(2Ridge+2LGB)이므로fit숫자는N3pipeline회수와내부estimator수를구분한다. 각parameter/thread/defaultsampling도등록한다.

## 진단·해석

e0는한날내공통anchorerror,es는차감shape−trueΔ0. `MSE=mean(e0²)+mean(es²)+2mean(e0*es)`는대수항등식이며항들이직교/독립이라는뜻이아니다. e0²평균은전체행가중과날균등이complete24에서같아도queryday지원/NaN조건을명확히한다. rawshape대차감shape오차/최종0시RMSE/일평균편향은사전서술진단으로유용하나여기서좋은부분만새판정문턱으로고르지않는다.

두머리를나눈다고level병목이자동해결되지않고0시라벨오차는shape훈련target전체에전파된다. 이위험을기존H1/H2반례와함께보고한다. 성과가미달이면y0head/shape0subtraction명세의추가효용미확보일뿐모든state분해불가/전문가설명오류/전체W40GS효과없음은아니다. 구성원통과도후속freshBASE/PFN/context/layout전체등록이필요하다. 원9모델유지/새답안0/정확재현미해결범위는변하지않는다.
