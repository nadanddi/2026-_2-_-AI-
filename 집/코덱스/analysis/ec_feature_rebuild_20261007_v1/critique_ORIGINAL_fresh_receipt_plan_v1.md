# 원 fresh receipt 소스 사전 비평

2026-10-07. validator/probe/registrar1 및 기존 producer3/PFN2/feature-loader2의 인코딩 소스를 읽었다. 등록·실행·모델/정답/채점/GPU/worker변경0이다.

## 판단

fresh66 준비를 선행하고 DIAG0 raw81/PFN5·6 계약을 실제feature/label/context SHA·trainmedian과 대조하는 방향은 타당하다. **실제 import 함수 identity를 강제하는 조건은 새 버전에서 보완한 뒤 감사 실행하는 것이 안전하다.** fresh reconstruction은 독립model refit도, wholemixed/SG2 causalgate도 아니라는 범위표시는 맞다.

### OFR01 — 등록 소스와 imported 실행 함수 연결

probe는 validate_raw/validate_pfn/verify_fold를 direct import하고 기대local validation1/completion2 파일을 pin하지만 실제 imported 모듈.__file__가 그 파일인지 대조하지 않는다. 새 검증모듈도 raw_producer/pfn_rules actual 경로를 자체 pin과 명시 연결하지 않는다. 현source fileSHA 확인만으로 다른sys.path 모듈에서 가져온 함수identity를 증명하지 못한다.

새probe에서 validation/completion/features를 modulealias로 import하고 __file__의 기대HERE path와 등록 source SHA를 검증한다. validator의 raw_producer/pfn_rules/baseline/checkpoint actual binding도 기대등록과 연결한다. productionfeature loader는 ownSHA/requireall66을 이미 검증하지만 이것이 다른 직접import helper까지 보호하는 것은 아니다. registrar가 실제runtime를 capture하지 않는다면 probe가 사전읽기/재구성 전에 직접 검증해야 한다.

## 인코딩·계약 수용

matrix_sha는 C-order little-endian float64 copy 및 NaN=np.nan 재대입으로 feature-loader2 matrixSHA와 맞는다. numeric_sha의 같은canonical 방식은 producer3 imputer SHA와 맞는다. y는 finite라 canonical NaN 영향이 없고 little-endian dtype<f8의 raw3 actualySHA와 같다. PFN float32 X[ix]/Q의 nativebytes와 y[ix]nativefloat64는 producer2 방식 그대로이며 sys.byteorder=little을 확인한다. 이는 소스 동등성 판단이고 actualmatrix 숫자 비교 실행 결과가 아니다.

rawvalidator는 row-ID index 정확순서/행수·fresh ySHA·완전expectedcontract/runtime/modelparams·median/allmissing/effectivecolumns·저장5error/3prefix 및 outputbyte 재읽기를 검증한다. PFNvalidator는 actualRNG2000 context재생/train-only membership·freshfloat32 matrix/selectedySHA·등록/준비/pred audit SHA·rules의120trace/21checks/4cache 구조와 byte 재읽기를 요구한다. 반납 predlist는 rawstage 숫자이며 shrink/clip/SG2를 적용하지 않는다.

registrar는 두driver source집합을currentSHA 및 충돌거부로 merge하고 rawreg/PFNreg/complete66/통계 및 새helper/source를pin한다. probe는 raw81foldmanifest를 먼저 확인하고 productionloader의24matrix 재구성 뒤 각contract를 대조하며 종료시 source와raw81hash를 재검산한다. 라이브workerfit함수나producer.main 호출은 없다.

## 남은 한계·강화 권고

새helpers의 FAIL negative selftest는 아직 없다. 등록/후속wholegate 전 matrix/y/median/contextID/column/index/producercode/audit tamper 거부를 pure/synthetic으로 시험하면 재개검증 주장에 도움이 된다. registrar의 각driverstatus/producerregistrationSHA·일정/count를 직접 검사하고 audit	source receipt의 recordedcodeSHA를 대조하는 것도 명확성 권고다.

freshfeatures는 같은기존 builder를 재호출해 저장SHA와 비교하는 replay다. 다른feature알고리즘이나독립model fit이 아니다. imputer directtrainmedian은fresh수치재구성이지만 raw3도np.nanmedian을 쓴다. PFN trainedweights/cache내용의 입력값으로부터 재생한 증거가 아니라 저장fingerprint·trace·inputSHA를 검증한다.

현재probe는 DIAG0/81raw/PFN2에 제한된다. 66fold/PFN264에 일반화하거나 elapsedworkerfitcount를 추정하지 않는다. storednumeric guards·3sampleprefix는 실제modeloutput의 전체행causality 재검증을 대신하지 않는다. storage manifest와validation본체는 소비한blob에SHA를연결하나 전체폴더atomic snapshot은아니므로 wholegate에서완료immutable결과와동일소비bytes를pin한다.

실행등록·실제probePASS 뒤에도 resultwholefalse/heldoutscorefalse를 유지한다. 원24/고정통계·최초미사용1회·최종의미있는선별정리파일검증 이후goal종료 의무는 남는다. producer/currentdriver source는수정하지 않고 새probe/validator버전으로 보완한다.
