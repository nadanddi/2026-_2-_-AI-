# SG2 probe2 실제 결과 독립 감사

2026-10-07. 결과2·등록1/2·registry 및 현SHA를 읽기 전용 검산했다. 모델/실SG2 실행·평가정답 수치·성능·GPU 실행0이며 부모worker 변경0이다.

**probe1의 identity-only 한계는 probe2의 실제 활성144행 범위에서 보완되었다.** 결과2는 DIAG10fold4 exact960 row IDs/960 sourcechecks/max0, choice snapshot exact960 IDs/allcomplete/sourcechecks960/max0이다. active144/has_candidate144이고 day>=179과 active flag 불일치0이다. 선택한 모든 reference day의24시간 IDs가 같은farm fold train에 포함되고 현재queryday와 다른지 metadata로 검산했으며 위반0이다.

12fresh-cache poison row IDs는 사전등록한 각farm/pass1·pass2 첫queryday h0/6/23와 정확 같다. 모든choice_equal/fresh_cache=true/difference0이다. permitted/forbidden count는 samefarm earlierqueryday full24h+current0..h와 나머지query라는 원registry metadata 계산과 정확 일치한다. 소스가 그packet을 먼저 확보하고 나머지ctx._query를 교란한 후 fresh plan/emptycache에서 선택을 재생하는 점도 이전소스비평과 같다.

probe2 output→probe/source등록 SHA가 현파일과 같다. 등록2 source224핀은223 기본MATCH+pandas1 승인된 읽기SHA MATCH로 모두 일치한다. 등록1도 같은 pandas 승인SHA 근거와 나머지 현핀 검산으로224핀 일치한다. 두 source등록은 각각의 probe 버전을 보존하며 probe1 active0을 소급 수정하지 않는다.

## 증거의 정확한 범위

이번 결과는 **실제 reference input/train EC로 구성한 SG2 선택과 adapter/source arithmetic**, 고정synthetic predictionprefix .8+.001hour에 대한 onefold 검증이다. evaluatedquery EC는 숫자로 읽지 않았고 baseline/candidate model output을 입력하지 않았다. 따라서 성능향상·actualmodel mix 동등성·guard .30의 모든경계·모든66fold causal proof는 아니다. sourcechecks960 중816은 pass1 identity이며 실제선택 범위는144다. 참조 후보 선택144가 실제correction변경144를 의미하지 않는다.

fresh poison12는 future/다른farm 지정값77777에 대한 제한된 입력교란이다. 960행 모든cut의 전수교란, 모든missing/tie/nonfinite, input 접근 instrumentation의 증거와 구별한다. 선택 캐시를 fresh로 한 점은 source 누수검사의 필요한 보완이며 이것만으로 universal causal gate가 완성되지는 않는다.

새 핵심 probe blocker는 없다. whole_pipeline_gate_passed=false와 syntheticprefix=true는 맞다. 실제R3/PFN completion·엄격receipt·rawstage mix/shrink한번/clip/RAW_PASS SG2·원행 전수 source/currentlineage/독립 gate 뒤 정답parse와 고정통계를 적용해야 한다. 원전체24·최초미사용1회 및 사용자 요청 선별최종정리파일 검증 뒤 goal종료는 계속 미완료다. raw/PFN live·memoryguard는 부모의 운영관측이며 이번 metadata 리뷰가 contextfit완료수나 전체runtime resource를 독립검증한 것은 아니다.
