# 고EC 전용 단일클래스 탐지 · 일반일 포함 검증 계획

사용자 요청: 고EC만학습하되검증에서일반날을남겨오탐을확인. 앞서일반ET는class1만학습시score1로퇴화(6.474)했으므로고EC입력분포지지를학습하는OneClassSVM으로별도실험한다. 고EC기준일평균sub_ec>=1.2,같은농장당일0..h입력,원등록73prefix특징유지. 목표는분류,EC수치회귀0.

공개360일8640행(고EC26·일반334),고정DIAG10 전체360/기존FRESH7·EL1 각뒤구간46(고EC5·일반41). 기존등록27fold의학습에서high==1날만남기고검증에는high0/1모두유지한다. day24행통제/같은농장query±1·다른농장±3·소비40일주변제외를동일하게유지,학습고EC는fold13~22일실제기록. median imputer keep_empty_features→StandardScaler→OneClassSVM(RBF,nu=.1,gamma='scale',tol=.001,max_iter=-1)을고정,첫2DIAG중간검산후나머지25fold실행. 모델·nu·gamma·threshold후선택0. 모든고EC날이24행이라학습행균등가중은날균등가중과같다. sampleweight따로사용0. 알고리즘은난수미사용/결정적이므로동일3seed중복fit을독립시드증거로제시하지않는다.

imputer/scaler/SVM은해당fold고ECtrain에만fit,일반일정보를학습전처리에도넣지않는다. 73캐시특징은각날현재prefix독립계산임이이전독립재검산으로확인됐고sourceSHA로고정한다. 정답과기록번호자체/다른농장입력/미래입력feature0. 공식train_y값추가읽기0·소비40점수0·test0·이전대회자료0. 최종400일모델생성/제출0,기존모델교체0.

SVM의native predict+1은고EC분포내일로해석,+1을고EC판정으로고정. decision_function은보정확률아닌경계점수,probability라고부르지않는다. 주h15/보조h0·8·23 각날1예측으로TP/FN/FP/TN·고EC재현율·일반오탐률·정밀도·balanced accuracy를보고. 시각을합쳐독립일로세지않는다. 학습은각시각행전체의인라이어비율/서포트벡터수기록,학습과검증의시각차이를명시한다. 같은날당일24시간평균고EC분류이지순간EC문턱이아니다.

비교는always-high(고EC재현100/일반오탐100),always-normal(재현0/오탐0),기존동일fold ET양class분류기(.5고정)저장ensemble. ET비교는가족/학습범위가함께다르므로고EC-only학습의인과효과로읽지않는다. nu/gamma검색·오탐보고후문턱조절0. p검정/채택판정0,기존노출360일/세검증동일뒤46일로새홀드아웃확증아님.

모든data/code/plan/baseline27CSV SHA고정. plan→첫2fold mid→full27fold final 독립비평,키와dayfold/모든trainhigh/일반train0/행무교집합·현재입력경계·전처리fit범위/최종counts를재계산한다. root는CSVloop로confusion·비율·precision 등핵심3숫자이상다른방법검산한다. 새이름파일·own폴더만쓰고타AI폴더읽기만. 최종결론은주로오탐과미탐지수를표본분모와함께명시하며신뢰도산술높음/새로운날일반화낮음이다.
