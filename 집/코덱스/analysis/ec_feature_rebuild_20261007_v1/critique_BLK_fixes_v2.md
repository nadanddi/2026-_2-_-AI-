# BLK 비평 보완 확인 v2 — 2026-10-07

판정: **요청된SG2 prefix/컬럼순서와endpoint공식/불변성·통계명세의주요보완을소스와감사파일에서확인했다. 비평전체종결/최종baseline PASS는아니다.** 신규fit·정답채점·GPU없음. PFN live상태는부모전달사항이며본검토로완료라표현하지않는다.

- adapter A01: v2 predict_one이baseline prefix의정확한0..h IDset과finite를검사한다. audit v2는비상수시간별baseline을사용하고source와1440행최대차2.22e-16,48future/otherfarm검사를기록한다. constant.7만으로pm오류를감출우려는보완됐다. 실제R3/PFN출력과최종pipeline 감사는남았다.
- adapter A02: v2 twin은실제WV MultiIndex의선택컬럼을직접순회하고A/b차원을assert한다. sortedW정렬가정을제거했다. no-finite/firstrecord 합성edge검증(A03)은현재v2파일의제한에도남아있다.
- methods M01/M05: endpointv2는block/anchor수·querydays·정확한flank끝ID·trainmembership·referencefull24·finite를assert하고깊은referencecopy를MappingProxy로봉인한다. finalcandidate clip도명시적으로구현했다. boundaryv2는원ID시간에서weight를독립계산한1440공식검사를기록한다. baselinepipeline와blend위치/clip순서가등록되었으나실제통합감사는남았다.
- methods M02: loss=seed별SE의행별평균,blockSSE/count,rowweight,farm별4block재표집,draws20k/RNG/ties/plus1 p/alpha.025÷6가명시됐다. 기존validator의endpoint선택정의가없다는문제를인정하고BLK진단전용·전체채택금지로제한했다. M03/M04는EC similaritycomponent명칭과coverage/distance/component진단을요구하여과장해석을억제했다. 이것은그래프시간방향/절대근접성문제가해결된것이아니라한계를명시한고정가설이다.

## 아직 수정할 pin 결함

**BLK_method_registration_v2의method_code_sha256에는blk_sg2_refonly_v1.py만있고현재사용할blk_sg2_refonly_v2.py가없다.** 새baseline/scorer runner가v2를쓴다면currentadapterSHA를실행계약에추가해야한다. 부모의새runnercontract가v2를별도로포함하면그증거로닫을수있으며기존등록파일을덮어쓰지말고새version으로보완한다. sourceaudit_v2자체는v1/v2양해시를이미기록한다.

statistics.CI는2.5/97.5 percentile의개별95%CI이며6variant보정CI가아니다. p판정의alpha보정과혼동하지않고'개별서술CI'로표시해야한다. 기존selection문자열의전validator개선은향후조건이고현재cross_validator_status=BLKdiagnosticonly가적용된다. scorer의기계상태가이를명확히구분해야한다.

다음은PFN완료확인→현재v2adapter의의존pin→실제raw혼합/one-shrink/clip/SG2/endpoint전체출력인과감사→BLK진단채점이다. 아직TM/P2LOO/EL1후속endpoint명세·최종1회봉인·전체채택이남았다. 초기BLK수치로기초자료정보부재나과거단서실패원인을확정하지않는다.
