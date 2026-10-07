# BLK 3방법 독립 비평 v1 — 2026-10-07

판정: **방법3개 공식·임의 고정 계수·train-only 참조·prefix 입력경계는 명확해졌음. 제한된 경계 진단 PASS로 진행 가능하나 baseline과 통합 판정 명세를 먼저 보완해야 정식 효과 시험이 가능하다.** blk_endpoint_methods_v1.py, BLK_method_registration_v1.json, register_endpoint_methods_v1.py를 읽었다. 신규fit·성능 채점·GPU 실행은 없다.

## 확인한 강점

PAST=0.8baseline+0.2왼쪽23시EC, BOTH는동일혼합에양끝EC원record-hour보간, GUARD는동일온실train-onlyEC mutual-best graph와현재일0..h 원14열 평균/현재값기반배정이양끝과일치할때BOTH이고그외baseline이다. query/gap 정답을읽는코드없고CSV직접접근없다. 배정 입력assert는같은farm·현재시각이전으로제한하며현재일만사용한다. ratio·margin·alpha가점수보기전임의가설임을명시한것은타당하다. 양끝역할/원record시간보간의한계를명시했다. 현재경계audit는constant.7을사용하여모델성능과분리한다.

## M01 · P1 · baseline 이름과변형6만으로는실험봉인이완성되지않음

BLK_QUERY_ROLE/RAW_PASS 이름은있으나 R3/PFN raw/clip/shrink/SG2순서, SG2fit_reference/refcalendar/role/빈경로처리, endpointblend가SG2전후어디인지와최종clip이봉인되어야한다. predict는받은baseline을혼합한뒤clip하지않는다. 이규칙을나중에결과보고바꾸면새variant다. 특히SG2 prepare의pass1query통계fit 위험은이method모듈밖에남는다.

필수조치: baseline spec SHA·단계출력·적용범위·referenceIDs를고정하고 endpoint는어떤단계출력에혼합하는지명시한다. source/input의현재SHA와fullpipeline 인과감사를통과한뒤fit한다. 현재경계PASS는이gate를대체하지않는다.

## M02 · P1 · bootstrap 통계와기존검증기 적용규칙미정

등록에는 'seed-mean squared error'만있어 (평균seed예측−y)^2인지, seed별SE평균인지불분명하다. 길이5/10블록의재표집에서행가중SSE/count ratio인지블록동일가중인지도미정이다. 모든seed×검증기동일개선이라고하면서 TM/P2LOO/EL1에서각방법이참조할고정양끝·gap/anchor정의가없다. 기존validator를BLK모양으로바꾸면다른검증기가된다.

필수조치: bootstrap의정확한loss수식·row/count weighting·farm층화여부·p이상/초과·ties처리·CI·RNG를코드에고정한다. 기존3validator는원분할유지하고, 거기에endpoint방법을어떻게적용할지누수없는anchor선택함수를사전정의한다. 불가능하면BLK효과진단만판정하고기존3검증기통과/전체채택은보류한다. 8blocks는독립가정이강하며샘플1440독립행처럼표현하지않는다. 20kdraw p는MonteCarlo오차를가지며0도진짜0이아니다. 6variant모두장부와보정에포함한다.

## M03 · P2 · graph component는시간방향 사슬과다름

mutual-best는각record의입/출연결을최대1개로제한하지만뒤날짜제약·cycle거부가없다. 낮은EC변동기록은과거↔미래양방향연결/순환이가능하며union은방향을버려component로압축한다. 이것은EC endpoint similarity component이며실제동의연속시간사슬을증명하지않는다. 현재root양끝일치는단순동일component조건이다.

조치: 명칭을component/연결가설로제한하거나true directedchain을원하면새version으로cycle거부/허용시간방향을사전고정한다. 링크수·component크기·cycle수·양끝samecomponent block수·guard activecoverage를점수와분리해진단한다. 희소연결로guard가전부fallback이면실험무효가아니라해당규칙coverage0결과다. 사후threshold완화하지않는다.

## M04 · P2 · 상대거리margin이절대유사성을보장하지않음

nearestcomponent distance는그component내최소record거리이며큰component가더많은최근접기회를갖는다. 차순위대비10%차가있어도query가모든trainrecord와멀수있다. finitecoverage75%는입력가용성이지근접/같은동보증이아니다. 이거리만으로assignment accuracy를보고하면안된다.

조치: 현재고정방법그대로시험할수있으나 component크기·절대1/2순위거리·margin·유효차원·시각별coverage를보존한다. 사이즈교정/절대threshold는새후속variant이며점수보고즉석추가금지. 주어진기작전체무효/정보부재확정으로확대하지않는다.

## M05 · P2 · 구조assert·reference불변성강화

layout blocks와anchor를zip하므로길이불일치때조용히잘린다. 같은farm만assert하고anchor가해당block의flank끝/start인지검사하지않는다. layout과source가봉인되어현재맞을수있지만범용runner오염검사로는부족하다. reference dict를그대로저장하여후속변경이chain/scale/pred간불일치를만들수있다.

조치: block/anchor길이·querydays일치·양끝ID정확도·endpoint trainmembership·24h완전성·finitebaseline/labels·입력/라벨immutableSHA를확인한다. leakageassert에gaplabels/queryinputs 직접참조금지,sourcehashpin,최종outputfuture/otherfarm/order/singlequery를추가한다. boundaryaudit의BOTH검산은같은calendar_weight를재사용하므로별도원시ID시간에서weight를직접재산출하는검사가필요하다.

## 과거 단서와차이해석

6.345~348의앞날선정/정답사슬에BLK양gap삭제·동시8block·고정양끝을추가한것은새검증이다. 다만등록의6.347은경계사례분류기록이지직접chain대조가아니므로해당source를명확히매핑한다. 새성공/실패는원recordendpoint보간·고정alpha.2·현재prefix componentguard에한정되며, 다른출처/진짜달력/더강한미세시간모형/전체기초자료의가능성을판정하지않는다.

우선 M01/M02를 실행 전 보완하고, M03/M04는 제한·진단을 보존한 고정 가설로 시험한다. M05의 오염거부/독립수식검사까지 연결한 새버전으로비평피드백을닫는다.
