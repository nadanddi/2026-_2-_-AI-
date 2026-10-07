# 도메인24 raw strict receipt·SG2 선택 고정·조합 독립 비평

2026-10-07. domain_raw_receipt_v1.py/domain_sg2_plan_v1.py/assemble_domain_BLK_v1.py 및 실제 연결 SG2 source를 읽었다. 모델 fit/GPU/보류 정답 값 열람/성능 채점 0. 부모가 실행 중인 모델을 중단하거나 재fit하지 않았다.

## DPOST01 — assembly loader 스키마 불일치 (P1, 조합 실행 전 수정 필수)

assemble_domain_BLK_v1.load는 모든 input record에서 `hidden_truth_loaded`를 읽는다. 기존 BLK_PFN_CPU_REFONLY_v3/context5.json과 BLK_R3_v1/LGB_seed47.json의 실제 키를 독립 확인한 결과 이들은 `heldout_truth_loaded`만 가진다. 도메인 새 raw는 `hidden_truth_loaded`다. 따라서 첫 PFN load부터 KeyError가 나며 현재 assembler는 실행 불가다.

새 버전에서 파일유형별 expected key를 명시하여 False를 검증하거나 두스키마를 엄격 정규화한다. key가 없을 때 defaultFalse로 통과시키지 않는다. producer 출력과 원 모델은 보존하고 새assembler/pipeline source pin만 등록하면 되며 모델 재fit은 필요하지 않다.

## SG2 선택 고정의 타당성

SG2 source의 reference_day는 공개 reference calendar/입력 서명·현재까지의queryprefix 및 reference distance로 선택된다. 실제 코드를 확인하면 chosen이 정해진 뒤에만 baseline_prefix 평균pm과 .30 correction guard가 사용된다. p/pm을 selection에 사용하는 분기는 없다. 따라서 zero prefix로 reference_day/level/has_candidate를 관측해 고정한 뒤 실제 prefix로 unchanged `.5*(level-mean)` 및 `.30` 조건을 적용하는 것은 source상 타당한 계산 최적화다.

query마다 own current/past 입력을 reference-only 통계에 transform하여 선택하는 작업은 query로 통계를 fit하는 것과 다르다. 모든query의plan을미리만들더라도각choice가ctx.query_prefix(rid)로만 계산되며apply는자기rid choice만읽는다. 원calendar/signature 스케일·reference EC levels는 학습참조에서고정된다. 별도query정답/새weight/threshold선택은없다. 미래query를 reference로 fit하는 path는 발견하지 못했다.

RAW_PASS pass1은 has_candidateFalse이고 identity다. query role의 nofinite/노후보 역시 그대로fallback이다. active값은 저장되지만 apply는 has_candidate만 사용한다; source상inactive→has_candidateFalse 계약과 부합한다. zero prefix에서의 changed 상태를 재사용하지 않고 실제mean에서guard를다시평가하는 점이 중요하다.

source equivalence는 baseline8640행 전수와 candidate72×2scope×48probe를 확인하고, future/otherfarm poison에서는 cached choice만재사용하지않고원sg.predict_one을재호출한다. 이는 단순cachedplan invariance를source인과검사로오해하는 문제를피한다. 현재는 검사 코드이며 실제 PASS는 아직미확인이다. 후보 전체1440행에 대해 원source를 재실행하는 검사는아니므로 한계를 유지한다.

## raw receipt의 독립 median·완결 검사

full mode는 exact72 raw,3baseline replay,complete의exact75filename집합과현재SHA를확인한다. registration/source/환경/ID/matrix/pred 및finite1440/3batchdifference/실효imputer열/alias도대조한다. np.nanmedian을학습5520행에서재계산해producer sklearn imputer statistics SHA와비교하므로 train-only 중위수에독립산술근거를추가한다. 저장SHA값이맞는다는것은fit전체를독립모델재실행했다는뜻은아니다.

읽은 partial receipt는13candidate raw/1baseline replay/차이0.0에대한PARTIAL_NO_SCORE_GATE였다. 전체72/3재현완료로해석하지않는다. 새snapshot파일이생겨도기존partial receipt의범위는늘어나지않는다. fullverify_raw()를assembler가호출하므로partial을scoregate로올리는우회는없다.

median SHA는NaN을포함한bit기록이라다른구현의NaN payload 차이로비수치실패가발생할가능성은있다. 지금의partial은일치했고허용오차를완화할이유는없다. 실패시수치median차이와NaNencoding차이를구별하여기록하고,새정규화정책은사전등록한다. sklearn버전·drop-empty칼럼검사도유지한다.

## 조합·감사 범위 및 등록 전 추가 메모

candidate mix=.6*(.6ETcandidate+.3fixedLGB+.1fixedMLP)+.4fixedPFN, shrink1회,reference EC min/max clip,SG2,clip이다. endpoint보정은추가하지않아새가족효과를조건부로비교한다. seed별전1440행shrink를fsum으로대조하고SG2에는currentday0..h predictionprefix만전달한다. 미래memberprediction은API에없다.

pipeline registration은아직없으므로그source목록의완결을본리뷰가확인하지못했다. 등록에는originalcachedPFN strictreceipt/gate/member files와R3fixedmember/oldassembledbaseline/audit,원sg2post.py,SG2adapterv1/v2,plan/assembler/rawverifier,rawregistration/complete/receipt 등transitive근거를전부묶고실행전후현재SHA를확인해야한다. 이번loader자체는원PFN/R3의registration/receipt를재검증하지않으므로pipelinepin이나freshbaselinehelper로해당근거를보완한다. plan source_sha는adapterv1 하나만 가리키므로optimizer 전체sourcepin으로오해하지않는다.

최종verifier는6baseline tag/24candidate exactmap×3seed×2scope/ID1440/finite/allrawSHA와실제source/check수 및plan semantics를독립확인한다. choice/reflevel/active/hascandidate snapshot을source/inputs SHA와함께기록하면선택최적화의실행재현이명확해진다. 이는새학습이나문턱선택이아니다.

현재명백한실행차단은DPOST01이다. 이를새버전에서닫고pipeline등록이완성된뒤실제assembly/최종인과gate를실행한다. raw모델진행을중단할새누수는발견하지못했다. 전체24원TM/P2LOO/EL1,54탐색대조와전체196구분,미사용seed/layout최초1회는그대로남는다.
