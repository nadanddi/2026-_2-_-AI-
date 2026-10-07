# 원66fold 도메인 raw runner 초안 독립 비평

2026-10-07. `original_fold_features_v2.py`, `run_domain_original_raw_v1.py` 및 model factory source를 읽었다. 아직 fit등록/실행0인 초안이고 모델·query·score를 실행하지 않았다.

## 직전 OFM01/02 수용

production API는 False probe flag를 노출하지 않는다. fit등록status·materializer 자체SHA·모든fit sourceSHA·completeSHA를 검사하고 load_fold(...True)를 호출한 후 require_all66=True를 강제한다. runner의 fit_one도 True와 loaderSHA/completeSHA를 fit 전에 다시 확인한다. 이 생산 경로에서 probe bypass 조건은 닫혔다.

materializer는 baseline/context/domain/checkpoint actualmodule.__file__를 own folder와 비교하고 환경version을 확인한다. runner도 materializer/model factory 실제경로를 확인한다. fit등록이 materializer를 포함하도록 API가 요구한다. OFM02의 핵심 linkage는 반영됐지만 runner에서 직접 import한 domain/checkpoint2/threadpool/library의 실제 binding은 일부 indirect경로에 기대므로 최종 등록에서 runtime/module path를 정리한다.

## raw 실행 전 필수 보완

**OR01 — resume imputer receipt 검증.** freshfit는 ET/MLP median을 train-only np.nanmedian과 exact비교하고 allmissing/effectivecolumn을 기록하지만 resume는 그 `imputer`를 전혀 검사하지 않는다. 같은 matrixSHA와 prediction digest는 saved imputer 내용이 현재train과 맞는지를 보장하지 않는다. 재fit 없이 현재 train에서 medianSHA/empty/effectivecols를 재계산하여 saved내용을 검증한다. LGB imputer={} 정책도 정확히 강제한다. audit의 fit_rows/query_rows·reverse/scattered/prefix diff 배열의 길이/finite/nonnegative/각tolerance 및 summary max 일치도 검사한다. 현재 max값 하나만 보는 방식으로는 선언한 fullresume 감사가 완성되지 않는다. 이 지적은 새학습성능 오류가 아니라 재개gate 결함이다.

**OR02 — 전체완료 시 재개 종료.** 모든5346개가 이미 있을 때도 fold별재검증 후 최상위complete에 `assert not complete.exists()`로 실패한다. partialresume는 가능하지만 completedresume가 정상no-op/검증종료하지 않는다. 기존complete를 exactexpectedfilename집합/현재SHA/등록/66및594/4752와 비교하고 같은 경우 완료로 종료하는 경로를 추가한다. 잘못된 기존complete를 덮어쓰지는 않는다.

**OR03 — fit등록 및 통계/후처리 경계.** 현재 필요한 등록파일이 없고 runner는 그status·seed·counts를 요구하므로 지금 fit실행하면 안 된다. 등록에 exact66fold키/train/query/forbidden,24family exactcolumns, complete66/396+filepins, modelfactoryactualparams, trainlabels/inputmatrixSHA, environment/CPU/thread policy 및 이runner/materializer/coredependencies를 봉인한다.4752=66×24×3과594=66×3member×3seed, 총5346은 맞다. 그러나 그count만으로 검증모집단·손실·판정·다중비교 정의는 고정되지 않는다. 원TM111는DIAGfold가생성한예측 중 고정111일subset이라는 의미와 P2LOO/EL1 전체합산,seed별방향·DIAG10p/Bonferroni 및후속ensemble/SG2를 성능조회 전에 연결한다. rawstage의R3-only 산출물을 full .4PFN 기준선 성능으로 채점해서는 안 된다.

## 추가 검사와 현재 장점

full66 completion과 각fold preparedpayloaddigest/exactIDs/24freshmatrixSHA를 확인하며, model선택은 고정ET/LGB/MLPfactory다. ET600/LGB800/MLPearlystop 정책은 source factory에 있고 fit은 train배열에만 수행한다. imputer/scaler도 Pipeline의trainfit이며 queryfit가 없다. futurequerypoison 후 prefixfeatures와 fullmatrix를3query/fold exact비교하고 reverse/scattered+3prefixpred를1e-6으로 확인한다. sampledprefix 범위를 전수query 인과증명으로 확대하지 않는다. prepared source와 MASK/recorddisjoint 경계는 직전materializer비평대로 별도 전제다.

singlewriter O_EXCLlock은 source/token으로식별되고 정상finally에서자신의token만삭제한다. 다만 lock파일 생성·jsonwrite가try밖이므로 그사이실패하면stale lock이남는다. explicitPID/starttick 확인 후 복구하는절차를 정하고 다른livewriter lock은 제거하지 않는다. 현실적인 compute량(5346fit)을 진행하기 전에 checkpoint재개와complete 처리의부정사례를 모델fit없이 확인하면 좋다.

현재 fit_one signature의 labelSHA는details로받으며 m.fit에는인자y를쓴다. main이바로tr.sub_ec에서y를만들어 현재불일치경로는없지만 계약을강하게하려면 fit직전 actualySHA/길이와detailsSHA를비교한다. 모든sourcepin이재확인돼도 y인자자체의일치검사는별도다. 새receipt에는actualmodelparams/featurewidth·효과열 및 source/runtime지문을 기록하면 후속baseline재현에 도움이 된다.

이 초안은querytruth채점/PFN/fullpostprocess가없으며 complete status도 이를명시한다. OR01~03을등록/새버전으로정리하고actualcomplete66fresh검증후에만rawfit을시작한다. 도메인BLKrank·고EC구간효과로원24후보를제외하지않는다.
