# 원 완료 manifest guard 초안 독립 비평

2026-10-07. guard/audit/current 저장receipt·SHA만 검토했다. 모델/실행worker/평가정답/채점/GPU 작업0이다.

저장완료≠수치/model/fullgate 경계는 docstring과 반환status/wholefalse에 분명하다. raw66/5346·PFN66/528 exactmanifest/폴드manifest합치 및 파일byteSHA를 확인하려는 설계는 타당하다. 그러나 **'path안전'·'동일byte에 대한 manifest 검증'이라는 강한 계약으로 fullgate에 연결하기 전 두 보완이 필요하다.** 현재 정상registry/output에서 실제 침해나 저장오류를 발견한 것은 아니다.

## OCM01 — Windows path·link 경계

PurePosixPath의 상대2parts/정규화 검사는 Windows drive를 검사하지 않는다. 예컨대 C:/a.json은 POSIX에서 absolute=false/2parts/canonical이므로 caller가 같은 expected_names를 주면 verify_manifest 입력검사를 통과할 수 있다. default파일 reader가 Windows Path로 조합할 때 drive/ADS 의미를 가질 수 있다. 또한 root/folder/output의 symlink·junction은 정규 문자열만으로 root밖 탈출을 막지 못한다.

새 guard에서 ':' 및 Windows drive/root를 거부하고 expected fold validator/number의 정확타입·allowlist를 좁힌다. 실제 root/folder/complete/output 각각 resolved path가 intendedroot 아래인지 확인한다. safety를 readcallback 이전에 요구하려면 lexical 검사와 physical reader의 resolved containment를 분리해 수행한다. 현재 rawregistry가 sourcepinned라 이 입력이 생겼다는 뜻은 아니며 generic guard의 경로안전 주장 결함이다. synthetic에 drive/ADS·linkescape를 추가해야 한다.

## OCM02 — hash·parse 및 시간 범위

complete를 sha(path)→read_text→sha(path)로 읽는 것은 parse한 bytes와 저장 beforeSHA bytes가 동일하다고 엄밀히 보장하지 않는다. 새 guard는 complete bytes를 한 번 읽고 그bytes의SHA와 json parse를 같이 만든다. source외부 변경과 파일교체가 없는 작업이라도 이 계약이 더 직접적이다.

output 파일은 차례로 한 번 hash한다. 다른fold를검사하는 사이 먼저 검사한 output/complete가 바뀌면 finalrootSHA만으로는 탐지하지 못한다. storage snapshot PASS는 그 순간들에 읽은bytes의증거이지 atomic 폴더snapshot이 아니다. actualwriter완료/immutable source등록을 선행하고 futurefullgate가 실제 소비하는 동일bytes/hash를 대조하거나 끝에서전부 재검산해야 한다. worker진행중부분감사 자체를 fullgate로 사용하지 않는 현재정책은 적절하다.

## 검증한 실제 증거와 누락 범위

합성10unsafe는 readcallback전 ValueError, tamper1은 byteSHA 불일치로 거부한다. actualfirstfold81/hash·foldcomplete를 저장receipt와 현SHA로 다시 대조해 불일치0이다. audit/guard/raw등록3 currentSHA도 모두 일치한다. 현재rootcomplete는 없다. incomplete66 refusal은 root파일부재를 확인한 것이며 fake rootcomplete가 있는 상태의 부분fold·extrafold·global/fold disagreement negative까지 실제 시험한 것은 아니다.

exactmanifest는 예상한fold모든81 또는8json의누락/추가를 거부하며 strictlower64SHA, rowmodelstatus및 heldfalse를 검사한다. 다만 root의extra fold directory/비json 또는nested output을 탐지하는 guard는 아니다. producer record안의rowIDs/matrix/y/runtime/audit수치·cache lineage, physical trainbounds, reference/SG2, predictions causal은 별도producerstrictreceipt/fullgate가 필수다. registration_sha 인자는 어떤등록을 뜻하는지 자체검증하지 않으므로 caller가current등록SHA/sourcepinned exactregistry를 검증해야 한다.

반환 'ORIGINAL66_COMPLETION_MANIFEST_PASS_NOT_FULL_MODEL_GATE'는 storage범위에서만 타당하다. 원5346 raw/PFN264 fit완료나 실성능독립검증을 이guard 하나가 증명했다고 말하지 않는다. 전수matrix/source/producer receipt·수치·SG2·혼합 causal gate 전 정답parse 금지, 원24·고정통계·최초미사용1회/최종선별정리파일검증은 남는다. pinned producer 수정 없이 새guard/감사파일로 보완하고 후속 sourcegate에 pin한다.
