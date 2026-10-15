# 첫 fold 산출 및 감사 수정 독립 검토

2026-10-09. audit_v2/run_v2/verify_v1 정적검토와첫DIAG10fold0 NPZ/JSON·registration을무학습독립검산했다. EC수치정답로드0, 새fit0.

## 첫 fold 독립 실제 검산

- NPZ SHA 및 metadata recipe_sha가 등록 recipe의표준정렬JSON SHA와같다. train8304행346일/query960행40일,각ID유일및불교집합. queryday는등록fold0와맞고훈련날은동일농장모든검증day±1밖에있다.
- 원/훈련외입력MASK 특징비교 receipt 1,478,112셀은8304×89×2조건과일치한다. 실제feature값을재생성하지않았으므로이항목은root assert통과receipt검토다.
- foldweights빈도는 .04계열23/.2 2015/1.0 6266로합8304이다. .04는binaryfloat표기로0.04000000000000001이며잘못된새계수가아니다.
- season의두농장모든anchor_day는실제훈련키에속한다. selectedrawkey/표준화전체fit계열은코드에서훈련keys로제한되어있고query외기를전달하지않는다.
- BASELINE/DROP5/MED3세저장배열은각960개유한값이며NPZ rowIDs는metadataqueryIDs와전수같다. 첫fold만으로세검증기후보성과나진행PASS를평가하지않는다.

## 코드수정 검토 및 남은 실행차단점

audit_v2는전체T=24×중앙평가day+5/17로hit.t>T와take.t<=T를검사한다. 이전지적의허용과거변조에의한거짓FAIL가능성이해소됐다. 아직전체audit실행완료receipt가없으므로현재부분통과와전체PASS를구분한다.

run_v2는NPZ SHA와metadataqueryID전수대조를추가했지만 **recipe source_sha=현재run_v2파일SHA로기존registration_v1의run_v1 SHA와달라진다.** 기존reg==recipe 및cachedrecipehashassert가실패하므로등록호환성처리가필요하다. root에게즉시전달했다. 원model/feature/통계설정불변확인후기존등록recipe를그대로보존하고실행runnerSHA만별도receipt에기록하는방식이타당하다. 첫fold를새설정으로변한것처럼표기하거나registration_v1을덮어쓰면안된다.

verify_v1은공식sub_temp만수치접근해fold출처/날buffer/anchors/독립fsum RMSE/20k bootstrap/그룹통계/스크린판정을별도로계산하도록되어있다. 훈련원label검사와EC숫자정답접근은없다. metadatajson의일치만으로전체rawmappingfit키전수증명은아니나코드키선택과교차연결하면충분히범위를설명할수있다.

전체계속조건: 등록호환sourceSHA처리를새버전runner에서명시하고전체audit_v2 완료를확인하면진단계속가능하다. 첫fold실제산출의무결성/shape/purge/가중치증거는통과했다. 이승인은CODEX구성원두후보screen범위이며전체W40GS검증/채택/제출승인이아니다.
