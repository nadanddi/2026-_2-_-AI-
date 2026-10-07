# 원domain 통계 등록2 source 독립 비평

2026-10-07. `original_domain_statistics_v2.py`, plan1, 실제statistics등록2를읽고source와draw파일hash만새로비교했다. 모델·정답·통계함수·draw재생·합성검사를실행하지않았다.

## 현재 봉인 수용

현재5개sourceSHA mismatch0, drawfileSHA 일치다.34,800,000bytes=200000draw×87block×uint16 2bytes다. farm별block42/45, DIAG8640uniquequeryID·TM2664행, membership8640개 및rowcount합8640/2664를등록metadata에서확인했다. reject_zero_TM_denominator=0은actual저장된실행기록이다. 독립draw재생/각count벡터검사는아직예정이므로이미독립PASS했다고쓰지않는다.

plan의offset0 floor(originalday/5)·nonemptyblock모집단·TM동일membership·farm별자기block수만큼복원추출·전후보/DIAG/TM shareddraw·rowweighteddenominator와source가맞는다. key정렬후membership을고정하며8640queryID중복을거부한다. TMdenom0이면loss읽기전에metadata로재추출하므로결과기반draw선별이아니다. 이번저장draw에는재추출0이지만향후synthetic에서그분기를확인해야한다.

SEED2026100703·200k·plusone/ties>=0·ALPHA.025/84는source상고정돼plan과일치한다. bootstrap_loss는row별seedmeanloss를인자로받으며blockloss합을sampledrowcount로나눈다. TM밖block을없애지않고sum0/count0로원모집단안에두는설계다. samplefile의정확길이/EOF·denompositive를확인한다. 명목daybin을물리적5일chronology라고부르지않는limits도적절하다.

## 중요 미완료: future score gate

**OS01 — 함수docstring은접근gate가아니다.** `bootstrap_loss`의‘AFTER full model gate’는설명뿐이다. 함수는drawfileSHA/layoutdigest/등록source/runtime/fullmodelgate를검사하지않고어떤same-size draw파일과layout이든받는다. 현재봉인산출물의오류는아니지만이함수를production score API로바로사용할수없다. future scorer/productionwrapper가truth수치해석전에actualfullmodelgate·statistics등록2·현재source/drawSHA·layoutdigest와정확DIAG/TM ID집합을fresh검증해야한다. synthetic전용함수와productionentry를분리하면경계가명확해진다.

**OS02 — 교집합·seed방향 판정은아직구현되지않았다.** 이source는draw생성과subset별p/CI만반환한다. 두subsetp모두<.025/84와각3seed×TM111/P2LOO/EL1의strictdirection,24고유candidate전체,adoption=False는plan에만있다. future score코드에서정확히강제하고score조회전에sourceSHA를등록한다. 두p중작은것선택·유리한seed선택·BLKrank제외로바꾸면안된다. DIAG/TM중첩을독립두성공으로표현하지않는다.

**OS03 — draw/layout 부정검사.** 현재bootstrap_loss는길이만같은다른count벡터의farm합42/45·허용index/width/layoutshape를자체검증하지않는다. draw현재SHA검증과독립rawcount재생이이를보완해야한다. original/source/draw/plan등록이sourcegeneration전후모두불변이었다는closure와futurefreshcheck도필요하다. 실제등록에는Pythonruntimefield가없고main이version3.12를assert하지않으므로현재binarySHA봉인은강하지만‘Python3.12generatorruntime까지검증됐다’고확대하지않는다. rawfit등록또는statsloader에서actualruntime/sourceimport경계를연결한다.

## 예정 independent/synthetic 검사의 필수 범위

원foldregistry와TMmetadata에서독립적으로farm/dayblock·행수·membership·full24·duplicate를재구성하고layoutdigest를비교한다. 모든200k binarycount를독립randrange흐름으로재생하여farm합42/45·nonnegative·TMdenompositive·SHA/정확bytecount를검사한다. 같은원draw를읽는것만으로독립generator검산이라고부르지않는다.

zero rowdelta는p1/CI0, positive는p1, 모든rownegative는p1/200001과negativeCI가돼야한다. DIAG/TMsubset정의와rowweighted정규화·TM외zero기여를각기확인한다. 키누락/추가·nonfinite값·손상/truncated/extra-byte draw·wrongSHA·wrongfarmcount·잘못된layout길이및zeroTMdenom은실패해야한다. math.fsum의극단finite입력overflow/nonfinitebootstrap출력도failclosed인지확인하면좋다. 본source는aggregatefinite를따로검사하지않으므로future실제loss의finiteguard를유지한다.

현재draw등록자체를무효화할core오류는발견하지못했다. OS01~03은futureproduction score와독립감사를닫는조건이며fit/score완료증거가아니다. 기존BLK54·상위최종미사용확정규칙은그대로유지한다. BLK노출후만든prospective탐색이며새독립확증·전역누적FWER증명·채택으로표현하지않는다.
