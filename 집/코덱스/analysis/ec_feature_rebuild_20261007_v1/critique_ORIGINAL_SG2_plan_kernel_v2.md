# 원SG2 plan1·커널2 소스 비평

2026-10-07. 두 초안·합성audit·RefOnlySG2 실제 소스만 읽었다. 실SG2 생성/선택/모델/정답/채점/GPU 실행0이다.

**커널2는 float PFN context key 거부를 추가하며 고정 산술은 그대로다.** 합성384Decimal/max2.220446049250313e-16/poison10/invalid22는 실제SG2 또는 fullgate가 아니다. SG2 adapter 합성5scalar/max0/12거부 역시 Oracle을 주입한 계약 검사다.

원SG2 소스에서 reference 후보선택·calendar/signature distance는 baseline 값과 독립이고 마지막 a1-prefixmean guard만 baseline에 의존한다. 따라서 plan1이 zero prefix로 선택을 freeze하고 같은 .30 guard/.5 correction을 actual prefix에 적용하는 분리는 소스상 타당하다. day<179는 choice조차 실행하지 않고 identity를 반환한다. query samefarm/currentday prediction0..h exactordered key·scalar, rawscope, distinct training ref를 요구하며 choice는 MappingProxyType로 노출한다. 실제source를 audit_source로 다시 대조하는 경로와 source snapshot도 있다.

## 전체 gate 전에 필요한 조건

1. `snapshot(require_complete=True)`는 choice전수 유무만 강제하며 `_source_checks`는 단순 횟수다. 같은row 반복으로 횟수를 채울 수 있으므로 실제 source-equivalence 전수 증거는 row별 검사집합/후처리 seed·candidate·prefix 계약을 별도 기록해야 한다. 5scalar 합성으로 원모집단 모두 검증됐다고 주장하지 않는다.
2. **future-input 교란 감사는 fresh plan/새 choice cache로 선택을 다시 계산해야 한다.** 이미 freeze된 choice의 apply만 교란 전후 비교하면 입력을 다시 읽지 않아 무조건 동일하고 선택 자체의 누수를 검사하지 못한다. original sg2 predict_one 및 fresh plan.choice/currentprefix를 함께 대조하고 앞query fullpast·현재h·다른farm/future 경계 consumption을 기록한다.
3. ctx/reference/scales/graph/labels source lineage는 외부caller가 증명해야 한다. self.sources는 선택된5개 핵심소스이고 context/baseline/numpy/runtime 등 전체전이 pin을 대신하지 않는다. 실제matrix/producer receipt와 completefold 및 reference IDs/labels/MASK를 원조립등록에 연결한다.
4. `_choices`와 `self.sg` 및 `self.sources`의 내부상태는 외부Python code에서 변경 가능하다. MappingProxy는 반환dict의 값을 보호하지만 plan 전체를 보안 격리하지 않는다. 구조·source currentSHA·choice digest를 조립 gate에서 확인한다. snapshotstatus/wholefalse 자체를 채점허가로 쓰지 않는다.

비차단 권고: audit_source에도 flags 정확bool/선택day의 integer scalar검사를 맞추고, registry순서·complete24h인 ids를 caller가 명시검증한다. synthetic에 inactive/candidate없음/no-finite/tie/missing·guard .30 경계/선택day가query 또는다른farm인 실패를 보완하면 actualflow 전 한계가 더 분명해진다. 현재 .8→.9 Oracle은 실제 weather/anchor 선택 동등성의 증거가 아니다.

실제 source 선택/수치0..h/forwardfuture·다른farm audit 전까지 SG2 plan의 whole causal PASS나 전체기준선 재현을 허용하지 않는다. RAW_PASS day179 gate·커널 mixed/shrink한번/clip은 prospective 원recipe이며 oldWT2 all-TM/GPU 동일성 주장은 유지할 수 없다. 기존 producer들은 수정하지 않고 새assembler/등록이 초안 및 audit source를 pin해야 한다.
