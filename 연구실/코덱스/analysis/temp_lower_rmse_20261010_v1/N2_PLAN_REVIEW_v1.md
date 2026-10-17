# N2 타농장 증분 prior 계획·코드 독립 비평

2026-10-10 · 연구실 · 코덱스 critic. PLAN_v1/N2_PLAN_v1/n2_lofo_v1.py 읽기만. 새fit0/EC수치0/공식자료외접근0.

판정: **시간당형태prior의LOFO적합성검사는타당하다. 비유한·출처증거보완후진행가능.** 현재코드는타농장leave-one-farm-out이고대상F13/F47/F32를처음부터제외한다. 제외농장센서·정답을학습에전달하는경로는없다. 이것은타농장분포내일반화gate이며대상F13/47이득의검증이아니다.

## 시간쌍과 지원 분모

farm/day/hour정렬후같은farm/day차분과hourdiff1, h>0,finiteΔtarget/Δair를요구하여자정연결·비연속행이섞이지않는다. pairtarget에라벨차분을쓰지만predictor에targetlag/farm/day/rowID는포함하지않는다. current14+diff14+hour주기2=30열로현재입력을활용하므로수준정보가전혀없다는주장은피해야한다.

eligible≥96쌍은훈련전지원정의이며분모는그조건의농장전체다. 각농장LOFO성능을본뒤분모를재선택하지않는다. ≥10farm/1000pairs 그리고≥80%farm의two-baseline strict개선과farmmacroMSE strict개선이명확하다. 96쌍은최대약5일분량에서도충족되므로충분한계절/관리다양성을보장하지않는다. 결과에농장별일수·hour분포도남기면지원범위를해석할수있다. 지원부족은효과없음이아니다.

## preprocessing·weights·NaN

LOFOfold train에서median/empty0/Scaler/Ridge를fit하므로heldout농장전체input을preprocessingfit에넣지않는다. farm균등weight평균1 공식 len(train)/nfarms/nfarmrows는Ridge loss에만전달한다. median/scaler는비가중이며행수큰농장의입력분포영향이남는다. 이것은고정설계선택으로가능하나'전체preprocessing도농장균등'이라고설명하면틀리다. 사후가중전처리변형을같은후보로추가하지않는다.

현재Δair유한필터가current/previn_temp유한을사실상보장하지만직접확인하면뜻이명확하다. 나머지센서feature에는NaN허용이나inf는imputer/Scaler가처리하지못한다. **학습전30열의비NaN항목finite를검사하고, predict/target/mse finite를assert하며JSON allow_nan=False로남기는보완을권고한다.** 계획의'비유한은개선아님'과코드의strict비교만으로는macroJSON NaN표기를완전히막지못한다. support부족case의prior배열NaN은의도된미예측이며성과계산case와구분한다.

## 출처·학습·산술 증거

recipeDATA SHA를원Pin과대조하는것은타당하다. runtime_v1모듈SHA·현재import __file__도연결하면DATA의RAW내용및sub_temp-only로더명확하다. 출력NPZ는rowID·farm·target·half_air·prior가있어독립pair재구성/농장MSE/farmmacro/80%분모검산이가능하다. 각foldtrainfarmkeys·train/querycounts·전처리모듈/versions·sourceSHA와eligible전체를등록/receipt에남긴다. 기준zero와.5Δair는실전모델baseline이아니라prior적합성의고정간이기준이다.

학습prior후실제target누적1열후보는별도등록이다. h0=0/dayreset은잠재수준을보존하는규약이지실제배지temperatureanchor가아니다. 중간Δ결측시나머지dayNaN·h23뒤reset/strictMASK/timeboundary등은target후보에서실제검사해야한다. prior는타농장전체로고정fit되더라도targetfold정답을사용하지않으므로그부분누수는없지만target의current/previousfeature추론세계는같은농장/현재과거만제한한다.

## 해석의 한계

LOFOgate에통계p/CI가없으므로80%/macroPASS를유의한대상효용또는시드강건성으로부르면안된다. 이것은계획상지원/적합성진행정책이다. 기존절대전이와Δloss/고정.5수축기각을단순재포장했다고단정할수는없으나타농장효용을target효용으로전환하려면fullW40GS새시드·문맥paired검증이필요하다. CODEX구성원만진단후성능개선완료로끝내지않는후속계획은타당하다.

원9CSV정확재현미해결과낮은RMSE목표는별개의평가다. .5127은실제제출플랫폼점수이며OOF가그보다낮다고실제다음점수가낮아졌다는뜻이아니다. 등록compatiblecache/실증probe/필요시새fullfit/전칸방향·다중비교기준을후속fit전에고정한다. 사용자제출요청없으므로새답안0 유지.
