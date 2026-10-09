# 고EC 양성일 LOOCV 독립 중간 비평 v1 — 2026-10-09

판정: 첫2일학습과예측의누수/자료범위/산술에는치명문제가없고등록된run_v3.py로남은24일을진행할수있다. 중간모형능력이나양성전체개선판정은하지않는다.

독립검산: critic_recheck_v2.py midpoint를직접실행해기존원자료10DIAGCSV및추가continuation sourceSHA,26개전LOO의복합farm/day purge/train/queryindex·classcount·query24행,첫2일48예측행/prior·3seed평균,24metricgroups/.5/.2탐지·positiveBrier/logloss·paired기준/gainloss/farm층화20k bootstrap를재계산해모두통과했다. 기존공개360일73features는이전분류기에서630720셀독립검산후고정SHA로재사용한다. 새정답/40일성능값로드는없다.

중간값: F13 120/121 두날모두h15에서.5와.2기준미탐지(0/2),LOOensemble평균score .0497051 대기존DIAG .1116571. 양성조건부Brier .9030897 대DIAG .7892466로 +.1138431 악화다. train296~297일/high20이며LOO라고항상고EC25일을학습하는것은아니다.

단일block문제: 두날은같은농장이고day//5가같아부트스트랩블록1개뿐이다. 재표집20k는그블록을계속반복하여CI양끝이같은+.1138431/p_worse1을만든다. 이는불확실성없음/통계적악화확증이아니라퇴화재표집결과이며독립증거0이다. 중간값을전체26일에확장하지말것.

발견된코드수정: 계획검토에서빈pool choice([],size=(1000,0)) 자체shape정상만확인했고다른농장의정수index와concatenate할때float로승격되는문제는놓쳤다. 최초score가그dtype로index할때실패했고v3는np.int64캐스팅만추가했다. critic도원v1 checker를보존하고동일int64·continuation검사를넣은v2를만들었다. 등록된8fit/2일예측은재학습·변경하지않았고순열조건/모델/threshold/원분할/자료pin도변경없다. 결과가좋아지도록수정한것이아니다.

남은해석한계: 양성only조건부loss는constant1을좋게평가하므로일반일오탐/전체분류유용성은측정하지않는다. 새LOOtrain에는원DIAG보다일반일과고EC가함께늘어나고일수/구성이같지않으므로왜차이가생겼는지를인과적으로분리할수없다. .2로임계값을사후선택하거나seed를고르지말고고정전체26진단만마무리할것. 최종범위에서도새최종400모델·보정·제출은0이다.
