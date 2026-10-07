# CH2 BLK 실행 전 등록 독립 비평

2026-10-07. 등록2·registrar2·runner2·query context1 및 기반 source를 읽었다. 현재 source hash와 후보 CSV만 재확인했고 query/model/score 코드를 실행하지 않았다. 도메인 조립의 approval credits 차단을 우회하지 않았다.

## 현재 확인

`CH2_BLK_registration_v2.json`의92개 source SHA를 현재 파일과 비교해 mismatch0이다. feature_candidates_v7은202행/202고유 candidate ID이고 처음196행은 v5의196행과 필드별 정확히 동일하다. 새6행은3방법×2scope이며 모두 UNTESTED다. registrar1 실패·부분 v6가 이 등록의 실행 증거를 대체하지 않으며 등록2와 v7은 별도 파일이다.

기존 whole-verified reference-only cached baseline을 재사용하고 모델 fit/GPU/domain assembler 호출 없이 input source 선택·고정 endpoint blend만 수행하는 별도 action이다. actual inference·gate·score 미실행 상태와 일치한다.

## 실행 전 보완 권고

**CR01 — 실제 query 모듈 import 경로 고정.** runner는 prefix/reference_loader의 재생 검사를 사용하지만 실제 `CH2QueryContext`를 제공한 module.__file__와 그 module이 import한 `blk_context_v1.__file__`를 확인하지 않는다. source92pins는 지정 파일의 현재 내용 확인이며 실제 import binding의 증거가 아니다. 미래 inference 실행 전에 own folder 절대경로를 assert하고 module SHA를 실제 audit에 기록해야 한다. Python -O 금지와 prefix 실제 경로 검사는 이미 있으므로 같은 방식으로 연결할 수 있다. 현재 shadow import가 발생했다는 증거는 아니다.

**CR02 — query loader의 전체 schema 선검사 선언.** querycontext는 expected ID만 직접 구성하므로 future/다른block/gap 숫자 접근은 source상 차단된다. 그러나 `query_prefix`에서 각 행의 schema를 확인하자마자 float로 바꾸므로 모든 행의 schema가 검사되기 전에 앞선 행 값이 변환된다. prefix3의2pass는 그 이후에 받는 이미 숫자로 바뀐 packet에 적용되므로 이 loader 경로를 소급해 보호하지 않는다. 등록의 ‘metadata/schema precheck before numeric conversion’를 전체 packet 의미로 유지하려면 loader도 identity/schema 전수 확인 후 별도 numeric loop로 바꾼다. 유효한 expected packet의 마지막행에 잘못된 schema, 앞선행에 ExplodesOnFloat를 둔 합성 사례로 경계를 확인하면 된다. 정상 원 CSV에서 key집합은 constructor가 RAW로 만들므로 실질 미래 누수를 발견했다는 뜻은 아니지만, 선언과 구현의 차이는 실제 실행 전에 정리해야 한다.

등록의 ranking 설명 중 ‘average11 common-reference input terms’는 최대11열 중 train 공통 가용·query finite인 term만 사용한다는 뜻으로 좁혀야 한다. 실제 분모는 결측·관측 길이에 따라 달라진다. pinned source가 정본이며 숫자 결과 후 가중/분모/문턱을 바꾸는 근거로 이 문구를 쓰면 안 된다.

## 규정·고정 정책

query context는 부모 BLKContext constructor를 호출하지 않으므로 부모의 전 query 수치 변환·양정답 로드 경로를 상속 실행하지 않는다. train_X의 query 문자열만 저장하고, train reference는 fresh loader로 받는다. exact same-block 이전 query day24h+현재0..h만 expected에 들어가며 gap/train 값이 query packet에 합류하지 않는다. query labels는 읽지 않는다. 문자열/파일 바이트 접근과 수치 사용은 구분한다.

prefix3에서 선택에 쓰는 값은 RAW input만이고 training EC는 topology와 양끝 reference 값에 한정된다. cycle component·missing/tie·one-sided·history disagreement fallback, immutable train scale, .2 endpoint weight 및 recordhour 가정이 봉인됐다. baseline은 이미 shrink/clip/SG2/clip된 값이며 후보 뒤에는 .8baseline+.2anchor와 train-bound clip만 있다. shrink/SG2 재호출이 없다. 원 historical submission14/GPU 동일성은 주장하지 않는다.

scope2×seed3×method3=18출력이며 statistical hypothesis는 seed평균 loss를 쓰는 method×scope6개다. registrar가6+48+6=누적60과 .025/60을 고정했다. domain의 이전54등록을 고치지 않았으며 새 단계의 탐색 예산에 미채점48도 포함했다. 이미 노출된 BLK에서 계산되는 이 p_worse는 탐색 screening이고 독립 채택 검정이 아니다. seed방향 조건·rowweighted farm-stratified block bootstrap·20k·tie worse·새 alpha가 명시돼 있다. 아주 작은 alpha에서20k Monte Carlo tail은 정밀도가 제한되므로 p를 과도하게 해석하지 않는다.

원 TM111/P2LOO/EL1과 각 fold train에서 graph/scale fresh 재생성이 요구된다. 최초 미사용 seed/layout 확인은 아직 별도 등록 전이다. 이 BLK inference가 원검증기 의무를 대신하지 않는다. 후보별 가중·문턱 튜닝 경로는 runner에 없다.

## 계획된 감사의 실제 범위

120개 probe×3method=360개는 forbidden query 문자열 교란 뒤 생성 packet 동일성과 선택 결과 역순 replay를 검사한다. 모든1440의 선택·18출력은 생성할 계획이지만,360은 모든1440×18 최종출력의 독립 미래 교란 검사가 아니다. source상 predict는 같은 선택과 baseline scalar로 결정되므로 합리적인 기능 검사이되, 실제 gate에서 조합 산술·baseline 정합·fallback·clip·lineage를 독립 검산해야 한다. query_prefix 로그는1440 기본 호출+120probe 호출로1560개가 예상되며 gate에서 중복을 무작정 오류로 취급하지 말고 역할을 구분한다.

현재 baseline preflight는 기존 gate와 조합 파일·cached receipt 및 전이 pin을 확인한다. source92 전체는 시작/끝에 검사하고 reference fresh replay도 끝에 재확인한다. 결과파일이 새 이름으로 쓰이며 score 실행 함수는 없다. 아직 `scoring_permitted=False`와 `adoption_permitted=False`이므로 이번 source 리뷰 자체로 gate를 열 수 없다. 실행 중 predictions만 저장되고 audit 작성이 실패할 경우 incomplete 산출물을 통과시키거나 기존 파일에 덮어쓰는 재개를 허용하지 않도록 후속 gate가 처리해야 한다.

CR01/02를 실제 실행 전 새 버전으로 정리하는 것을 권고한다. PF1/PF2 metadata 비교와 fixedflank 계보 설명은 포함됐지만 genericneighbor 파생 실험과 바닐라 원자료 inventory는 여전히 별개다.
