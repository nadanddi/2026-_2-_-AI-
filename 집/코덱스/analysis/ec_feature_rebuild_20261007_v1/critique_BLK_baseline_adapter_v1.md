# BLK baseline adapter 독립 비평 v1 — 2026-10-07

읽은소스: blk_baseline_data_v1.py, blk_r3_baseline_v1.py, blk_sg2_refonly_v1.py, blk_sg2_audit_v1.py, 배포SG2 prepare/ref_calendar/correct, R3등록·9raw출력·complete, SG2감사결과. 신규fit/정답채점/GPU없음.

## 현재 증거와 판정

R3 raw9파일을독립표준라이브러리검사했다. 각각1440finite출력·orderedqueryIDs·registrationSHA·predSHA가맞고등록의현재의존코드SHA도전부일치했다. train5520/query1440이며등록feature/calendar 감사는singleprefix6개+reverseorder1개총7검사다. 이는9raw출력보존/의존pin의PASS이며원모델전체동등성/최종인과PASS는아니다. ET/LGB/MLP가모델설정상배포규칙을따르고R3혼합가중.6/.3/.1을등록한것을확인했다.

SG2감사파일은5520ref-onlyfit/1440query/constantbaseline.7/source최대차2.22e-16/48future-otherfarm검사/활성1440/변경411/RAW_PASS활성0을기록한다. 파일·코드상한계는올바르게constant진단이라고표시되어있다. 부모전달session22417을직접poll했으나본agent에서는Unknown process였다. 이로종료/실패를단정하지않으며새worker를시작하지않았다. 부모가ownhandle종료를확인해야한다. PFN은완료라고주장하지않는다.

## A01 · P1 · 비상수baseline source대조와prefix완전성

predict_one은rid존재와baselineprefix가같은day의h이하인지만검사한다. 0..h모든hour가정확히한번있는지는검사하지않는다. 누락된prefix가오면pm=np.mean(values)는원source의일내expandingmean과달라진다. 현재constant.7대조는시간별pred정렬·누락·2중shrink/clip오류를드러내지못한다.

필수조치: baselineprefixIDset을현재record의0..h정확한IDset과대조하고finite를검사한다. deterministic비상수hour별/record별baseline으로원source대조를추가한다. rawR3/PFN혼합→oneprefixshrink→clip→SG2→clip의실제출력을대조하고단일query·순서·futureprediction교란을추가한다. 원source및adapter출력비교는query정답없이가능하다.

## A02 · P2 · twin컬럼순서가현재는맞지만하드코딩임

adapter b는sorted(W)×hour0..h이고source는WV pivot MultiIndex와hrs<=h mask다. 현재pandas pivot의알파벳변수/hour정렬에서는둘이일치한다. 그러나실제columns를보지않고정렬을가정하므로누락hour나pivot옵션/버전변화시순서/길이다를수있다.

조치: selected=self.S['WV'].columns[hrs<=h]를직접순회해동일(column,hour)순서로b를구성하며A.shape[1]==len(b)를assert한다. reference직접행을queryobs로넣은selftwin·변수순서shuffle·h0/5/23·일부missinghour합성검사로검증한다. source설정과달라지면새version으로등록한다.

## A03 · P2 · firstrecord/no-finite는원source버그완화이지전범위동등성아님

adapter known-prior없으면cq0, q유효차원0/finite거리0이면fallback한다. 원source는i-1wrap 또는빈거리argmin등으로실패할수있다. 현재BLK모든query에이전공개flank가있고외기완전하여이edge가실제감사에등장했다고볼증거없다. SOURCE-equivalent라는결론은감사1440행에한정해야한다.

조치: query가첫record/이전query외기allmissing/signaturezerovariance/후보0/유효차원0/거리모두nonfinite조건을합성검사하고명시적fallback값을등록한다. seed/모델효과로설명하지않는다. missing관측을0으로오인하지않도록count/finite의의미와NaN/Inf처리를검증한다. datetime이아닌recordday만으로선형시간이확정됐다고해석하지않는다.

## A04 · P1 · ref-only정책은적절하나최종전체경계는남음

SG2prepare는ref_inputs만받아pass1weatherfit을고정한다. mu/sd재계산역시ref-only다. refcalendar는trainref만사용하고시간별SIG표준편차도samefarmref-only다. queryfull_date는earlierquery에만허용하며futuretrainendpoint참조와futurequery차단을구분한다. 현재querysignature는0..h만쓴다. 따라서v1의querypass1통계fit위험을보완하는방향은맞다.

하지만SG2_sourceSHA는m생성시에만저장하고실제predict직전pin검사나reference불변성검사는별도runner에필요하다. 현재R3dataaudit는실제queryvaluefuture교란없이single6개/order만확인한다. 9모델출력single감사도첫8행으로한정되어있다. feature함수가인과적으로보여도최종미래/다른farm/순서/단일query감사가아직필수다. prepare_query가한꺼번에unionprefix를넘기는것은day내features가모두누적인경우에만안전하다.

조치: frozenrefcalendar/signature/weatherstatsSHA, R3/PFN각최종raw/혼합/후처리출력의경계감사와current sourcepin을추가한다. BLK_QUERY_ROLE baseline와RAW_PASS를계속분리하고SG2activation과changed를보고한다. ref-onlyfit은원naiveBLKprepare와동일정책이아닌의도적적합성보완임을기록한다.

허용다음작업은A01대조·A02/A03합성감사·PFNCPU완료확인·실제전체출력감사다. 성능채점/채택gate는그증거로열며constantSG2감사나R3raw9완료만으로열지않는다.
