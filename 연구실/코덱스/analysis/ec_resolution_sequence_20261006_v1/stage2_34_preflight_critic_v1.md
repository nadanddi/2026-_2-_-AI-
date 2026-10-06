# 단계2 위험학습·단계3/4 선행판정 사전 독립 비평

2026-10-06. PROTOCOL2_v1.md/clarification/stage2_v1.py/stage2_preparation_v1.json/stage3_4_v1.py/test_gate_v1.py 검토. 앞선단계1완료및실자료24CSV/50열화이트리스트감사결과를근거로사용했다. reviewer위험fit/성능열람/후보선택0.

## 사전 판정

**위험학습코드에서outer정답누수나등록모델·특징의불일치는발견하지않았다.** 본문에이미고정된후보의선행실행이가능하다. 판정코드에는아래risk진단flag경계의경미한문안불일치가있으며성능보기전에정합화하는것이좋다. 보정수식·선별gate의변경이필요하다는판정은아니다.

확인사항:

-stage2준비가드가단계1receipt·독립완료감사PASS·위험입력24파일SHA를확인하며38FULL/50design목록을서명한다.현재시각train.A−train.sub_ec>.15라벨만학습에사용하고query.sub_ec는입력feature에포함하지않는다.
-median(keep_empty_features=True)/std/L2LR(C1,lbfgs,tol1e-8,maxiter2000)변환/fit은outertrain에만적용하고query는predict_proba로만사용한다.각day24행의weight합1이며같은24행완전day조건이준비가드에서확인된다.0/1라벨한쪽만있는상수fallback도그constantlabel평균을사용한다.
-수렴경고는error로바꾸고maxiter한계도중지한다.기술적중단시완료receipt가없으므로후속stage34에진입할수없다.완료fit별median/mean/scale/coef/intercept/classes를저장하여예측산술을후속독립재현할수있다.
-단계3보정eligibility는act_vent_tdz≥.8/act_circfan_tdm<10의시점누적제어입력만사용한다.정답/high/day평균으로gate를정하지않는다.고정risk_core_v2의하향수식은risk>.8/최대.10/0하한이며LGB가A이상이면변화0이다.
-고EC정의는seed/farm/day실제평균≥1로사후채점에만사용한다.각seed전체/일반RMSE엄격감소,고EC무악화(숫자허용1e-12),pass2전체day≥179상대RMSE증가율≥2%면기각이코드와clarification에정확히고정됐다.고EC표본없음도보수적으로기각한다.
-seed평균후점수는참고출력이며통과판정은각seed다.같은날을공유하는3seed를독립표본으로세지않는다.SCREEN_PASS_NEEDS_FULL/SCREEN_REJECT모두adoption=false로보존한다.

## 개선·주의사항

1. **중간 — 위험진단flag의등호차이.** 원프로토콜의보고항목은risk≥.8인데risk_metrics는risk>.8이다.실제보정은>.8가맞지만진단선택건수는등록과다르다.성능결과보기전에진단flag만>=로맞춘새버전또는>=/초과두수를모두보고하는명확화가바람직하다.후보의threshold/cap/label/C나통과판정은수정할필요없다.

2. **중간 — 실제위험모델출력감사가남는다.** 사전fit0와입력자료PASS만으로새risk예측의누수없음/변환계산일치/수렴/통과가능성을선언할수없다.완료후fit12cell별원학습label/weight/median/std/coef로그를대조하고query위험점수를독립logit→sigmoid로재계산해야한다.메타기준source/prep/helper/inputSHA가현재서명과같음을stage34실행시독립확인한다.

3. **중간 — LGBanchor반례와상향위험은gate가자동으로해결하지않는다.** F13에서LGB가더높은사례는A−LGB≤0일때보정을막지만,다른실제고EC에서LGB가A보다낮은상황은계속과소를늘릴수있다.그래서고EC보호판정은필수이며공개141일의aggregate무악화가개별고EC무악화를뜻하지않는다.수정고EC행/날의SSE도보고하는현계획이적절하다.

4. **중간 — 선행표본은신규미사용확증이아니다.** 기존4사례/공개OOF분석을바탕으로후보를설정했고partial4fold는그공개영역과겹친다.프로토콜clarification이이를명시한다.이번에gate를통과해도전체nested/DIAG-A-B/현EC14비교·원채택통계조건을대체하지않는다.실패후같은스냅샷문턱·C·feature탐색은등록규칙대로하지않는다.

5. **낮음 — 기술중단과통계기각의상태를구별한다.** solver경고/파일중단은유효한SCREEN_REJECT성능표가생긴뜻이아니다.출력미완성·lock·traceback을보존하고해당중단상태로기록한다.원파일을덮어쓰기/partial성능으로완료표시하지않는다.

6. **낮음 — signature고정과git등록은fit전에완료한다.** preparation의source/helper/protocol/inputSHA와명확화/step34코드의사전등록기록을보존한다.현재risk학습의성능을보기전에고정한등록을이후coef나문턱수정의허가로확대하지않는다.다른AI의최신제출/클로드실험파일은이후보등록에섞지않는다.

현재코드는단계1진단이보정효용을보장한다고가정하지않고정답/분할메타를feature에서분리하고있다.진단경계를정합화하고고정서명을등록한후,사용자가승인한조건부선행실험을진행할수있다.후보성능PASS는실제완료결과와독립검산이후에만판정한다.
