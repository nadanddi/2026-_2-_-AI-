# 현재 EC14 확인과 주요 학습날 제외 진단

2026-10-07 연구실 코덱스. 사용자 “진행해봐” 승인. 제안한현재기준확인→학습날제외 진단을 순서대로 실행한다. 새모델채택·제출물은 만들지 않는다.

## 기준과 분할
- 기준코드: 집/클로드/submission14_ec_sg2/model.py와sg2post.py. R3는운영순서9특징을더한47/23열, PFN은이전38열과같은4문맥. .48ET+.24LGB+.08MLP+.20PFN raw혼합→.5현재+.5prefix평활→학습범위clip→SG2→clip.
- 원public DIAG10의10개fold×3seed(7/101/2024),총360고유날/8640고유시간행. 기존원R3캐시의train/query ID·정답/특징순서를사용한다. 이미검산한public y만읽고원train_y/test_X/EL1/잠금정답은새로읽지않는다.
- 모든SG2 ref와학습통계는해당fold의trainIDs에한정한다. query/잠금/±1purge 행은label ref에서제외한다. package의weather 표준화는실제제출에서모든pass1이학습자료이므로 training-only지만CV에서holdout pass1을fit에넣으면안된다. own sg2_ref_v1은prepare의pass1표준화집합을명시ref로한정하고나머지SG2규칙을그대로복사한다. 실제제출과같은전체training-ref 조건에서원prepare와동일함을검사한다.
- 검증은 **EC14 레시피의 fold내 재학습형 공개 검증**이다. 기존SG2C의연속평활·전역bounds/ref결과와혼동하지않는다. 실제test예측이나제출바이트재현을수행하지않는다.

## 1단계
현재R330그룹/90fit만재학습한다. PFN은동일train/query/features/context 서명검사후원캐시4문맥평균을재사용한다. 원데이터의operations와기존package features가같고현재·이전입력만쓰는지검사한다. DP1기존CSV는이미R3평활·clip된것이므로raw혼합기준에사용하지않는다. 새로운30raw멤버/clip前後/SG2前後와trainIDs/bounds를내local에보존한다.

360일검증이모두끝난뒤기존계절v2와현재EC14의대표4날·일반환기61/268·일반/고EC·pass1/2·온실별오차를확인한다. 손실이특정사례에집중되는지분모와함께보고한다.

## 2단계: ET 학습날 의존도
첫단계검토가끝난뒤다음고정2안을실행한다.

- D1: ET학습에서 F47_139의24행만제외.
- D2: ET학습에서 F47_139와F47_231의각24행만제외.

각fold의기존학습집합에해당날이없는경우그날은추가로제외할행이없다. 삭제행수와실제재학습유무를기록하며동일집합은baseline ET캐시를재사용한다. 그외ET설정/seed/학습특징/훈련season변환은고정한다. LGB·MLP·PFN출력,bounds,SG2의 label ref도고정한다. 따라서 **ET에서선택된날의영향만분리하는진단**이며학습자료전체삭제후새제출모델이아니다.

3seed×10fold를모두평가하고ET·최종의 변화를별도표시한다. F47_161의고EC지원/남은top날도추적한다. baseline/D1/D2의 원fold1seed7은 fulltree배열/훈련-ID/정답/입력/잎지원을저장한다. 삭제효과를고정forest기여와혼동하지않는다.

## 진단 판정(채택규칙 아님)
- F47_161에서세seed모두ET하루평균편향의절대값이줄면 **SELECTED_CASE_DEPENDENCE**로기록한다. 해당날만고친것과전체이득은분리한다.
- D2이후에도세seed모두ET하루편향>.20이면 **RESIDUAL_STATE_CONFUSION**도함께기록한다. 두판정은동시에가능하다. 새문턱선택은하지않는다.
- 전체/일반/고EC/pass2/각farm와선택161제외나머지날의RMSE·SSE변화를전부보고한다. 고EC악화/다른날손실이있으면삭제안을해결책으로추천하지않는다.
- 이번진단만으로새채택은없다. 가능한모델수정은별도등록하고원사용자규칙의모든seed×DIAG10/A/B방향·DIAG p_worse<.025(여러안조정)등을지켜야한다. 기존잠금/EL1재채점0.
- 단계3의특징영향제한/다양한학습날지원은이번진단을읽은뒤새가설로고정한다. 기존온실ID추가·h0전부삭제기각안을그대로반복하지않는다.

## 감사와 보존
캐시/source/ID/열/변환·label ref·정답 합산을 독립검산하고혹독비평을반영한다. source등록후fit,전체 완료receipt후score.기존파일보존·새버전만, 내analysis/local·main만사용. 다른AI파일/집worker변경0.

## 실제 앙상블 후처리 순서
세시드별 검증을 유지하되 실제 EC14 앙상블은 세seed rawR3를 먼저 평균한 뒤 PFN혼합·평활·clip·SG2·clip을 적용한 별도열로 계산한다. seed별최종예측의단순평균과는 SG2 gate와clip때문에 다를수있다. baseline과삭제안 모두 실제순서를따르고 두평균의차이도기록한다.

