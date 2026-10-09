# 정제 데이터·새 EC 모델 독립 계획/소스 비평 v1
2026-10-09 · 집 코덱스 · 후보fit0 사전검토
검토: PLAN_v1.md, run_v2.py, recipe_ec_v1.py, season_transform_v1.py, predict_model_v1.py, final_fit_v1.py.

판정: **정제 데이터 생성과 실험모델 학습 실행 승인. 확인된 차단 누수 없음.** 사용자 요청 모델까지 완수하되 성능판정과 채택/제출은 별개.

## 초안의 치명적 해석위험이 해소된 부분
q실제 일평균으로 불일치일을 제거한 조건부검증은 정답을 안 뒤 고른 모집단이며 실제평가에서 그필터를쓸수없다. 최종계획이 filtered 및 원q all을 공동필수/각alpha.0125로고정하고원qall/pass2all +2%guard를요구하므로조건부점수만으로실전개선주장하는위험은제한된다. 소스도같은q·같은filteredq를baseline/CLEAN모두채점한다. 두모집단의점수를직접비교한삭제이득계산은금지.

## 실제 소스검토 근거
1. load_public은 기존DIAG10공개8640행360일labels만 읽는다. CVfinalscore 저장전 원train_y를수치DataFrame으로load하지않는다. 원train_y SHA를핀하기위한바이트읽기는정답을특징/선정/채점으로쓰는행위와별개이다.
2. CV build_jobs는원x/y/f에서각oldfoldtrain/query를ordered선택한다. globalpublicclean목록/CSV는CVtrain선별에참조0. t/qt의동일옛matrixhash 및purge>=2를강제한다.
3. selector median/mean/std/season은삭제전outertrain기반. train분류에t만사용, q분류의q평균input/y는query_removed 채점표시에만사용. sourceclassify는t/qt같은함수라도q정답이훈련삭제를갱신하거나model로넘어가지않는다. 1회삭제고정, 자기·같은농장±1기록제외5NN, 일평균>=1/gap>.5, tie는distance/key로결정.
4. candidate는keep IDs의rawX와y에서features를새로만들고cleanrawX로season도재fit한다. ETimputer/MLPimputer/scaler/LGB/allbounds도cleantrain만사용. 다른PFN/SG2 경로를포함하지않는다. source47/Base_R3목록에target/day직접column없고prefixoperations/한시간RAW10특징은원causal형식이다.
5. baseline은원캐시ET/LGB/MLP를동일.6/.3/.1 및동일shrink/baselinebounds로재구성한다. baseline원season과candidatecleanseason차이는원자료vs정제자료전처리개입범위에속한다. 처음ET/LGB/MLP seed7의새factory replay도강제.
6. bootstrap은seed평균모델예측/farm×day//5 상대기록번호구간/20000 pairedloss. 각scope안동일rowIDs를BASE/CLEAN으로비교. 원rawlabels로HIGH층화할뿐model엔유입0. 원결과scores/runs출력은x모드이며중간비평존재관문이있다.
7. profile은원본/정제key정합·24시간완전·NaN/무한·범위·중복·음수target·날짜coverage를기록한다. 삭제는원rowID유지24행통삭제. selectorSnapshot에각fold행렬/스케일/y/이웃이있어새독립검산에서거리선정전수재생가능하다.
8. final_fit은FULL_DIAG10/모든10foldscore뒤원공식400일9600행을읽고같은selector1회→전체clean→9모델fit. 추가40일은최종학습자료로만소비하고queryprobe도기존public IDs만사용해40재채점/튜닝없다. final모델artifact와CVprediction은분리. 이는사용자요청에맞춘전체공식학습자료의실험모델이다.
9. finalpredict는frozenseason/저장9model/고정weights/bounds만사용하고testweather/평가통계fit하지않는다. modeljoblibSHA/codeSHA/reload/prefixmutation/delete/order/weathercheck 포함. 입력전처리selector를실제예측에적용하지않으므로q의숨은평균을요구하지않는다.

## 남는 위험/반론/최종 검증 관문
- ‘설명안됨’은47평균feature거리5NN의대리정의이다. proxy outlier 삭제로라벨오류/농업이상원인을확정하지말것. 정상분포를선호하는라벨선별이고실제high손해가능성은이전실험에서관찰됨.
- 정제검증qfiltered는QLABEL조건부모집단이다. bootstrap유의라도이모집단을실제평가에서미리선정할수없다. all공동기준/guard는계획대로결과가나쁜경우에도적용.
- 공개360globalcleanCSV와각foldcleanCSV, 마지막400fullcleanCSV는기준학습집합이달라서삭제day집합이서로다를수있다. 일치시키려CV에global목록을적용하면누수. 각자료역할/삭제ID/분모를모델manifest에정확히표시.
- 최종40일추가후모델의실제일반화오차는새로채점하지않는다.360CV결과는400finalmodel의직접성능검증값이아니다. 이미소비된40을다시검증으로쓰지않은점을그대로명시.
- finalprobe 각농장head2day는pass1만일가능성. 별도독립검증에서각farm×pass1/pass2의retainedpublic24시간일을포함하고.25/.75cut시같은날미래시간및타farm변조/삭제를실제predict_bundle경로로검사해야한다. 모델source변경없이새검증코드로보충가능. midcut만으로다양한시점에대한검증을완료했다고말하지말것.
- 모델save/load 및다양한inputshape전행동일성,manifest모든9model/seed/cleantrainingIDs/데이터SHA, replay3/후보90/final9fitreceipt, trainingRMSE↔holdoutgap/약한farm/pass층/소수날집중도를최종독립검산할것.
- run resume는이미있는foldcsv/meta SHA만1차검사하지만score에서registrationSHA도검사한다. source/cache핀변경과파일중복이없는지중간/최종에서다시확인.
- 노출DIAG/기존3seed의실험모델이지A/B·독립새holdout·새seed확증이아니다. 어떤점수라도채택0/새제출0/실력확증0. 실제모델파일생성은사용자명시요청에따라성능과무관하게완수.

전단계비평→첫fold독립재계산→전체CV 및최종9model실제predictpath독립검증의3단계로진행한다. 상기최종관문보충은실행차단이아니라결론범위와검증완료조건이다.
