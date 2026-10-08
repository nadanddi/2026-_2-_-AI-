# 실제 SG2 probe1·등록 probe2 독립 비평

2026-10-07. probe1/2·registrar2 소스, 실제등록2 및 probe1 저장결과/current SHA를 읽었다. 모델/SG2 실행·평가정답 숫자·채점·GPU 실행0이다.

## 현재 확인

probe1 결과는912 row IDs/912 source audit/max0/fresh-cache poison6을 기록하고 probe/reg 현SHA와 일치한다. 그러나 실제 choice active0/has_candidate0이다. **원day<179 identity 분기 검증이며 실제 참조 선택·guard 작동을 검증한 결과가 아니다.** 기존 모든행audit PASS status를 이 범위를 넘어 쓰지 않는다.

probe2 등록224핀은223개 기본읽기MATCH+pandas1개 정식승인 SHA MATCH로 전224개 현소스와 같다. registry상 DIAGfold4는960query, 그중day>=179144행이다. registrar는 양farm×pass1/pass2가 모두 있는 eligiblefold 중 최소fold가4임을 구조 ID로만 확인하고 source가fold4로 고정되어 있다. 성능이나 label수치로 유리한fold를 선택한 것이 아니다. 이번 검사 시점 **probe2 결과파일은 없다.** 실행 중이라는 부모보고와 계획만 기록하며 실제960행/active/선택/poison12 PASS를 선취하지 않는다.

## 수용한 설계와 한계

각 fold train/query/gap loader를 원registry로 고정하고 참조 ID/label membership이 train과 정확 같은지 확인한다. source audit는 고정 .8+.001hour synthetic predictionprefix로 모든 row를 한 번씩 exactordered ID 순서로 대조한다. source scalar 검산은 실제 model mixedprediction의 검증이나 성능선별이 아니다.

future/다른farm poison은 permitted query_prefix를 보존하고 나머지 query 값을 바꾼 뒤 **새 OriginalSG2Plan/빈cache에서 선택을 재생**한다. expected 기존choice와 newchoice·출력·원source를 대조해 이전 cached-choice만 비교하는 무의미한 테스트를 피한다. h0/6/23·각farm·각pass의 첫queryday12probes로 등록되어 있다.

전체144 active 가능구간 중 참조 후보 유무와 guard 실제변경 coverage는 결과 전 알 수 없다. probe2 완료 후 row별 active/has_candidate, 실제reference_day의train/samefarm membership, 144범위와양farm, poison12의 정확key집합/빈cache/prefix허용행수/sourceSHA 및 artifact 연결을 따로 확인해야 한다. source audit는 synthetic 한 prediction trajectory뿐이므로 .30 경계·여러 actualbaseline/candidate prefix를 모두 대신하지 않는다.

poison12는 유한 선정값77777에 대한 표본검사다. 원fold 전행 future-input·missing/nonfinite/keyschema 금지 및 source consumption의 보편 증명은 아니고 평가label 숫자를 poison한 검사도 아니다. 현재 prefix의 allowed ownfarmpastfull/current0..h와 reference-alltrain이라는 규정 경계를 fullflow감사에서 계속 유지한다. `all_row_source_audit=true` 등록은 source arithmetic대상 전행 뜻이지 causal교란을960행 전수 했다는 뜻이 아니다.

새 핵심 probe-source blocker는 발견하지 못했다. 이 audit registration/output은 whole_pipeline_gate_passed=false/no-score다. 실제 원5346 raw/PFN264·mixed/shrink/clip/SG2·독립 fullgate 전 정답parse 금지, 원24·고정통계·최초미사용1회는 그대로 남는다. raw 첫fold81complete/DIAG1 진행은 부모 actualpoll 보고이며 이리뷰에서 모델worker를 재관측하지 않았다.

사용자 추가 완료조건인 전체목표 완료 후 의미 있는 분석·조사·실험만 최종 정리파일 작성·그파일 검증 뒤 goal종료도 아직 충족되지 않았다. 중간 감사파일 누적 자체를 그 최종정리나 전체목표 완료로 처리하면 안 된다. probe2 actual결과 비평은 새버전으로 분리한다.
