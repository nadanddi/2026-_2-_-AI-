# OneClassSVM 고EC학습/양class검증 독립 계획 비평 v1 — 2026-10-09

검토: PLAN_v1/run_v1,경계지적반영PLAN_v2/run_v2. 판정: 일반일을전처리/fit에서빼고고EC입력지지를학습한뒤검증일반일을남겨오탐을측정하는한개의고정OneClassSVM 실험은타당하다. 27기존분할/73특징/nu.1·gamma scale/threshold를고정하고결정적알고리즘의중복seed증거를만들지않는범위도적절하다. v2의prepare+첫2fold진행가능하다.

자료/누수: 기존등록outertrain의high1인덱스만median/scaler/SVM fit에사용하며평가는원query의양class를유지한다. train과query24시간일그룹이교차하지않고원samefarm±1/otherfarm±3/잠금±1 제외를재사용한다. metadata의일평균target는train필터/평가정답에만쓰이고모델X는73현재prefixfeature만전달한다. 고EC일은hour별순간EC가아닌당일평균기준이며여러시각을독립일로합산하지않는다. feature/raw/27baselineCSV/코드/계획의sourcepin도좋다. 최종독립검산은실제hightrain복합키와모든queryrow/full시각을재구성해야한다.

발견수정: nativepredict+1과score>=0는경계0에서다르다. 비평가가실제설치라이브러리에공식데이터없이동일0벡터4개를넣어확인했을때score0/nativepredict-1이었다(단순라이브러리debug이며대회fit은아님). v2는native+1/score>0로정확히맞추고v1을보존했다. 이수정은결과후threshold튜닝이아니라실행전native정의정정이다.

확률/불균형: decisionfunction은고EC확률이아니고고EC학습분포지지의경계score다. inlier가실제고EC임을보장하지않으므로검증recall과FPR·정밀도·혼동행렬을같이보고해야한다. 누가고EC26이고일반334인지상수비율을모르면accuracy만으로오해할수있어balancedaccuracy와분모가낫다. nu.1은traininginlier정확90%를약속하는숫자가아니고실제traininginlier와supportvectors를기록하는것이타당하다. 학습all24h inlier비율을h15검증recall과같은모집단숫자로비교하면안된다.

인과/검증: 원양class ET와비교하면알고리즘가족·학습표본·전처리fit범위가함께변한다. oneclassonly로좋아졌다는인과효과를분리하지못한다. DIAG전체360/기존FRESH7·EL1의동일뒤46일은자료내대체분할이며새홀드아웃확증이아니다. 고EC도연속일에몰리고2차양성5일뿐이므로perfectrecall 등의정밀도를표본분모와함께해석해야한다. 현재고정SVM결과를모든oneclass기법의가능/불가능으로확대하면안된다.

이후: 첫2fold로label1train/전처리행수·featureSHA/query정합·baselinejoin/native경계/혼동수를독립확인한뒤나머지25fold를진행한다. 최종은27fold마다hightrain/전체query·4시각지표·farm/pass·alwayshigh/normal 및동일ET비교를다른CSV/math산술로검증한다. p검정·채택·threshold재선택·최종400fit/제출0을유지할것. 추가필수수정은없다.
