# N4 feature 감사 결과 상속 검토

2026-10-10 연구실 코덱스. v1/v2 source·registration·audit JSON과 `n4_audit_bridge_v1.py`를 읽고, critic이 별도 Python AST/해시/JSON 비교를 실행했다. feature 감사 재실행·모델 fit/predict·정답CSV 파싱0.

**판정: 현재 확인한 불변 feature 감사 범위에 한해 v1 실제 결과를 v2에 상속하는 방식은 정당하다. v2 전체 감사를 다시 실행할 필요는 없다. 현재 상속 receipt를 근거로 v2 최초 `--first` 진행 가능하며, 첫fold 뒤 독립 중간 비평 조건은 유지한다.**

## 실제 별도 확인

- `save/raw_fold/world/cx/folds/audit` 6함수의 `ast.dump(include_attributes=False)`는 v1/v2에서 모두 정확히 같았다. 앞선 전체 source diff에서 바뀐 부분은 PerformanceWarning 억제/첫fold 루프 제어/새파일 이름이었다. feature/audit 함수의 global recipe·ALPHA·IRR·해시계산식 AST도 같았다.
- 두 registration JSON의 다른 key는 `source_sha` 하나뿐이다. 모든fold의 ordered train/query ID, 원입력/checkpoint/runtime pin·feature·seed·threshold·분할/통계 설정은 정확히 같다. 첫query960/train8256 수가원등록ID수와같았다.
- 실제 v1/v2 현재source SHA가각등록값과같고, 현재 runtime SHA도두등록값과같았다. 부모 audit의원 recipe hash와bridge의부모 recipe hash가 canonical JSON 계산값과같았다. 새v2 recipe hash도같은규칙으로다시계산해일치했다.
- `parent_audit_sha`는원v1 audit파일SHA와같고 `bridge_source_sha`도실제bridge SHA와같았다. 원 audit의모든field는recipe_sha변경을제외하고v2에그대로보존됐다.
- 상속4,464,900셀 및변조검사 결과는v1 실행의증거다. 실제v1 audit 정상exit0은부모가전달한실행결과이고, critic은그feature함수들을다시실행했다고주장하지않는다. 이번별도검사는상속계약/출처/산술동일성을확인한13개항목으로모두PASS다.

## 왜 단순 파일명 변경이나 검사 우회가 아닌가

v1 감사를통과한검사대상함수/입력IDs/의존성pin이그대로인데,모델실행의`--first` 제어와결과파일명이바뀌었다. bridge는새source의전체감사실행을가정하지않고 **동일한검사대상함수에대한기존증거**를상속한다. JSON에 `UNCHANGED_FUNCTION_AST_EVIDENCE_INHERITED_NOT_V2_RERUN`와부모receipt/recipe SHA를명시해계보도보존했다. 기존파일덮어쓰기0이며새 source에맞춰recipe hash만전환했다.

AST일치만으로Python환경·모든외부상태의동일성을자동증명하는것은아니다. 현재는source/runtime·원입력/checkpoint와orderedID 계약이동일한상속이다. 모델시작때원package전파일SHA검사는여전히실행되며, 불일치가발견되면이상속을그변경환경의PASS로쓰면안된다. 이문서는그시작전검사를제거하는허가가아니다.

## 한계와 다음 확인

상속범위는첫DIAGfold feature union의LABEL/MASK·각농장첫query날h8 미래cut·타농장변조·gate합성검사다. weights/season/PFN예측의전범위인과성·배치독립성이나새전체모델결과를검증한것이아니다. v2감사를‘새실행PASS’로표기하지말고‘불변함수감사증거상속’으로표기한다.

다음실제첫fold에서현재등록/원source SHA·baseline726/9109 no-op·PFNcontextID/확률·weights/season·전체혼합항등식/g0 exact를독립중간검산한다. 단일PFN49 예비실험/신규layout 독립자료아님/사후6번째후보/채택0이라는이전한계는변하지않는다.
