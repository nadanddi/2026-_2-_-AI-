# 고EC일 한 날씩 제외하는 분류 진단

사용자제안에따라공개고EC26일을일단위LOOCV로검증했다. 고EC는일평균sub_ec≥1.2이며학습에는일반일과고EC를모두포함했다. 제외일24행을모두빼고같은농장±1/다른농장±3/소비40일주변도제외했다. 기존73prefix특징과ET3seed평균·logit/prior기준은그대로유지했다. 일반일은검증하지않았으므로오탐률·precision·AUC/AP·전체구분정확도를이결과에서계산하지않는다.

## 탐지 결과

|당일 입력 시각|고EC일|탐지(.5)|놓침|재현율|탐지(.2 보조)|
|---|---:|---:|---:|---:|---:|
|0시까지|26|6|20|23.1%|15|
|8시까지|26|4|22|15.4%|15|
|15시까지|26|10|16|38.5%|18|
|23시까지|26|11|15|42.3%|20|

## 기존 묶음 검증과 같은26일 비교

15시: 기존DIAG 8/26 탐지 → 새LOO 10/26. 새로잡은날4·기존에잡다가놓친날2. 고EC조건부Brier 0.521494→0.443512;이는양성날에만계산한오차로전체분류Brier와다르다.

각LOO학습에는 296~303일(평균298.12),고EC18~22일(평균20.08)이남았다. purge때문에항상나머지고EC25일을다학습한것은아니다. 기존DIAG의해당일학습고EC는13~20일이었다. 모든LOO학습집합은해당기존DIAG학습집합의superset임을training_exposure_v1.json으로확인했다. 일반일학습량도동시에늘어고EC표본수효과만분리할수없다.

탐색농장층화5기록번호블록 18개·20000회bootstrap: 양성Brier차이 CI95=[-0.1692505167715422, -0.00021453609393401812],p_worse=0.02445. 새고EC자료·미사용홀드아웃이없고training겹침·사건내날의존성한계가있다. 이값은자료내paired진단이며채택판정이나독립확증에쓰지않았다.

## 시드·농장·구간

- prior: 0/26 탐지, .2탐지0,고EC조건부Brier0.869877.
- logit: 9/26 탐지, .2탐지17,고EC조건부Brier0.457753.
- et_8383: 10/26 탐지, .2탐지18,고EC조건부Brier0.440785.
- et_1919: 10/26 탐지, .2탐지18,고EC조건부Brier0.448412.
- et_7171: 10/26 탐지, .2탐지18,고EC조건부Brier0.441980.
- ensemble: 10/26 탐지, .2탐지18,고EC조건부Brier0.443512.
- F13 뒤구간=False: 10고EC일,기존0→LOO0탐지,놓침10.
- F13 뒤구간=True: 3고EC일,기존1→LOO1탐지,놓침2.
- F47 뒤구간=False: 11고EC일,기존5→LOO7탐지,놓침4.
- F47 뒤구간=True: 2고EC일,기존2→LOO2탐지,놓침0.

## 해석과 검증

탐지수·학습일수의산술신뢰도는높음: rawCSV별도loop합계로각시각counts/mean/positiveBrier와pairedgain/loss를재계산했고, 독립비평가가원metadata/26purge분할/라벨·시드/실제예측파일/집계·bootstrap을검산했다. 실험설정변경은검증배치만,모델튜닝0/재균형0/특징변경0/새400일model0/제출0. 기존모델source/code/dataSHA와이번코드/계획SHA는매stage검사했다.

원인추정신뢰도는낮음: LOOCV로학습자료를더남겼을때예측이어떻게달라지는지볼수있지만, 자료가부족한지입력정보가부족한지모델설정이문제인지혼자서는분리하지못한다. .5에서놓친날도score.2이상인날이있을수있으며,보조임계값.2는사전고정한민감도보고만이다. 일반일의오탐손해를새로검증하지않은상태에서임계값을바꾸거나EC회귀gate로채택할수없다.

계획비평: 실제읽는10DIAG예측CSV의hash핀누락을v2에서보강했다. 첫2일fit후bootstrap에서빈F47pool float배열이연결index를float로승격시킨오류가났다. v3은연결index를int64로고정,원registration핀을유지하고continuation등록을추가했다. 모델·fit·저장예측은바꾸지않고중간집계만재실행했다. 중간은F13 120/121 단일block이라CI독립근거0. 최종비평내용은critique_final_v1.md에기록했다.

## 재현·자료

모든시드는8383/1919/7171. 학습은일sampleweight1/24·class_weight없음. 같은온실당일0..h RAW14/DP1/낮밤대비만features,정답·미래·다른온실현재입력제외. MASK불가5열은기존분류기와같이제외했다. 공식train_y값추가읽기0·소비40성능재점수0·공식test추론0·이전대회자료0.

PLAN_v2.md,run_v2.py prepare/first(첫2fit보존),bootstrap_repair_v1.md,run_v3.py score_midpoint/rest,registration_v1.json/continuation_registration_v1.json,local/predictions/26CSV+receipt,final_score_v1.json/local/final_paired_h15_v1.csv,training_exposure_v1.json,fresh_scalar_checks_v1.json,critique_*_v1.md 및critic_*recheck*가근거다. 재실행은기존폴더를덮어쓰지말고새버전폴더로등록해야한다.
