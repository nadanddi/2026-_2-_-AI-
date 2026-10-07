# 도메인24 등록 pipeline v2 독립 source 비평

2026-10-07. pipeline registration v2 / assemble v2 / verify_domain_BLK_v1 / score_domain_BLK_v1 / execute stage / SG2 plan / raw receipt를 읽었다. 등록100source SHA를 PowerShell로 직접 재계산하여 현재 불일치0을 확인했다. pinned 파일 수정·model fit·GPU·query EC 값 열람·채점 실행 0. 현재 raw fit 또는 조립 완료 여부는 이 리뷰에서 실행 확인하지 않았다.

## 핵심 판정

새 core pipeline 실행차단·누수·통계 계산결함은 발견하지 못했다. 이전 DPOST01은 domain raw의hidden_truth_loaded와원R3/PFN의heldout_truth_loaded를경로유형별로명시검사하여닫혔다. 누락키에defaultFalse를주지않고unknownfolder도거부한다. 실제fullraw/assembly/gate결과는아직대기이므로소스검토PASS를실행PASS로표현하지않는다.

## 계약 및 규정

executor는등록된4stage목록/실행sourceSHA/전체pin을실행전후검사하고직전stage 정상exit0·동일registration·sourceunchanged를필수로한다. rawreceipt는exact72/3replay/75filehash/currentmatrices/imputer실효열·독립trainmedian을확인하고partial을fullgate로사용하지않는다.

조합은ET가족만대체한.6*(.6ET+.3LGB+.1MLP)+.4fixedPFN,shrink1회,referenceECminmax clip,SG2,clip이다. 원baseline값을저장된fullbaseline과동일하게유지한다. SG2reference선택은referencefit통계와각rid current/past queryprefix에서추론하고예측mean은선택후수정단계에만사용한다. zero-prefixfreeze는queryfit이아니다.

assembler source-equivalence/future check수는baseline8640+72candidate-seed×2scope×48probe×2=22464다. future/otherfarm변조후원sg.predict_one을다시호출하여cachedchoice만비교하는공허한검사를피한다. 상위feature/imputer/raw모델source 및batch감사도연결돼야한다는scope를유지한다. 48probe는전체후보1440원source재실행이나무한input-domain인과증명을뜻하지않는다.

## 최종 gate

verifier는exact6scope-seedtag/24candidate/72stage/1440uniqueID 및baseline identity,모든stage·candidatefinite/clip범위/source·matrix/rawreceipt를확인한다. independent scalar3stage+2scope×72×1440=518400개mix/shrink/clip/SG2산술대조를예상한다.

SG2의불연속.30guard는원저장preclip와np.mean으로같은branch를고르고,선택branch안의수정산술은독립fsummean으로계산한다. guard를independentmean으로다시판정해threshold인접행에서정책을바꾸는것을피한다. 이것은원branch정책보존검산이며threshold를완전히독립구현으로확증했다는뜻은아니다. 원source동등성감사와함께해석한다.

gate는fit91pin+pipeline100pin+등록/assembly/rawreceipt/75raw출력·complete를절대경로transitive로묶어현재SHA를확인한다. scorer는truthfloat변환전에전체currentSHA·필수경로membership/등록stat·assembly/rawreceipt/code를검사한다. 이전IN01종류의stale-source gap을새계층에다시만들지않는구조다. source로부터계산한기록에대한검증이지모델독립재fit은아니다.

## 채점 통계

고정24×2scope48안+이전6=54,alpha.025/54,3seed전체RMSE개선,seedmean SE difference,8block farm층화row-weighted bootstrap20000,같은rng2026100702 및tie>=0/plus-one p를유지한다. 일반/고EC는24h정답평균>=1의score-onlysegment이며위치segment도고정됐다. 개별95CI는다중비교보정CI가아니다. emptysegment는None으로처리한다.

BLK는이미노출된탐색진단이며48안의순위/BLKscreenFAIL을원TM/P2LOO/EL1의24안제외조건으로쓰지않는다. 과거탐색전체196안및최초미사용seed/layout1회미완료도유지된다. 정답float변환과선별이모든gate후에있는source순서는적절하다.

## 성능·운영 한계 (차단 아님)

verifier는original_prefix구성에서ids.index를72×2×1440×평균12.5회반복하여불필요한O(N²)검색이발생한다. dictionaryindex를사전구성하면줄일수있지만현재pinnedsource수정은하지않는다. SG2원source대조baseline8640+candidateprobe6912 및future재호출6912도큰CPU비용이다. 느리다는이유로후보/probe/원검증기검사를줄이거나허용오차를풀지않는다. 필요하면새version에서동일출력·동일검사집합을입증하는구현최적화만한다.

stageexecutor는오류결과도실행receipt로보존하며동일stage재호출을거부한다. 실패후기존파일을삭제/덮어쓰기보다실패원인·정답노출여부를장부에기록하고새version/명시재개정책을사용한다. 원rawfit91pin은그대로보존되어현재진행중fit과후처리등록을섞지않는다.

현재등록source에대한진단조립·검증시도를막을새핵심사유는없다. 실제fullraw→assembly→wholegate→score순서의성공증거와점수독립산술검산은별도확인해야한다.
