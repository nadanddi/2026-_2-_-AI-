# 도메인24 실제 whole gate 독립 검토

2026-10-07. actual gate/assembly audit/execution receipts/원 등록 및 verifier·scorer source를 읽고 currentSHA를 확인했다. 모델·query·gate·정답 채점은 재실행하지 않았다.

## 판정 및 fresh evidence

제한된 등록48안 BLK 진단 채점을 막는 core 오류는 발견하지 못했다. 현재 gate180개 전이sourceSHA mismatch0, 기존 fit91개와 pipeline100개 pins의 gate membership 누락0이다. 이 수들은 독립적으로 읽은 source map 기준이며 두 등록의 합계가 overlap 때문에180과 같을 필요는 없다.

actual assemble/verify execution receipt 모두 returncode0·frozen_sources_unchanged=True·heldout_score_stage=False이다. assembly audit는72candidate/seed×2scope 및22464 source-equivalence/futurechecks, source차이최대0, independent shrink차이1.1102230246251565e-16을 기록한다. gate는518400 independentscalar·최대차이4.440892098500626e-16·whole=True·truth=False·adoption=False다. predictions/audit/rawreceipt/fitreg/pipelinereg/currentverifier/helper 연결은 source pin과 actual 기록에 부합한다. 이는 산출물 증거의 읽기 검증이며518400 산술을 새로 실행한 독립 score 재현은 아니다.

## 검증이 실제로 보장하는 범위

verifier가 verify_raw를 다시 호출하여 현재 원행렬·raw75파일·imputer audit·완료receipt를 연결한다. assembly의 exact1440ID·이전 cachedbaseline동등·정확6scope/seed×24candidate집합과 stage3개 finite/길이/clip을 강제한다.4cachedPFN 평균, R3 .6/.3/.1, 전체 .6R3+.4PFN, shrink1회, preclip, SG2 및 finalclip을 fsum 기반 별도scalar 식과 대조한다.

518400은72candidate-seed×1440×(rawmix/shrink/preclip3+scope2)다. SG2 reference 선택은 동일 고정inputplan을 재생하며, .30 guard branch는 등록된 np.mean 경계를 그대로 쓰고 correction 내부mean만 independentfsum과 대조한다. 원 정책과 다른 floating branch로 바꾸지 않은 점은 타당하다. 이 PASS는 독립 ET refit나 reference selection 알고리즘의 완전히 다른 구현 검산은 아니다. source-equivalence/sampledfuture와 전수 조합산술의 결합 범위를 유지한다.

gate의 전이 map에 후보raw파일/complete/assembly/pred/현재등록/source가 들어가고 scorer가 truth 숫자해석 전에 currentSHA와 핵심path membership을 검증한다. 새로운 caller가 gate 문자열만 읽고 truth로 넘어가는 설계가 아니다. 실제 부모 실행이 정상 Python 경로임을 유지하고 assertion을 무력화하는 -O를 사용하지 않는다.

## 통계·다음 경계

이번 score는 실행 전에 봉인된 domain48안+이전6=54, alpha .025/54를 그대로 사용한다. 이후 CH2 stage의60은 별도 신규 등록이며 이번 기존54를 사후 수정할 근거가 아니다. 전체 탐색 이력과 각stage의 판정 기준을 명확히 기록하고 두 단계를 하나의 새 독립 확증검정으로 합쳐 표현하지 않는다.

이미 노출된 pass1 BLK8block은 탐색 진단이다. all24원TM111/P2LOO/EL1 및모든seed와 최초미사용 확인은 남는다. 후보 rank/구간정답을 근거로 원검증기 대상24를 줄이면 안 된다. alias·대체변수 혼재, event co-change와 생리인과의 차이, historicalGPU/submission14 동등성 제한도 유지한다.

actualscore 및 독립Decimal checker/사후비평 전 점수 개선을 보고하지 않는다. 현재 gate를 수용하는 것은 고정된48안의 진단채점 경로에 한정되며 채택·확정구성·제출·바닐라 원자료 전수완료와는 별개다. 이전 approval credits 차단은 그때 실행되지 않았다는 이력으로 보존하고, 이번 실제 originalregistered action의 exit0증거와 구분한다.
