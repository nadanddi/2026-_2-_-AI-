# N5 v2 감사 후 실행 전 비평

2026-10-10 연구실 코덱스. v1→v2 전체diff, `n5_registration_v2.json·n5_audit_v2.json` 실제읽기와별도Python 해시/JSON검사를했다. critic N5 효용계산·모델fit0.

**판정: v1에서지적한finite/완전packet/결측관측/두labelnegative/게이트항등식은v2에반영됐고새핵심차단점은발견하지못했다. 부모가실제audit세션53470의정상exit0을확인한뒤 v2 첫fold계산진행가능하다. 첫중간검산전에전체계산하지않는다.**

## 코드와 실제 receipt 정합

- 공식raw 비결측온도 유한값/row_id서식assert가추가됐다. raw460농장일에각hour0..23을정확24개포함하는size/nunique/bounds검사와같은날q비증가assert를한다. 기존gate/threshold/formula/분할/통계량은변경없다.
- 결측→10℃→결측→7℃→12℃합성prefix에서q=[1,1,1,0,0,…]이며이전유효min10이현재결측에서도유지되는것을검사한다. sub_temp/sub_ec 각각열거부negative도추가됐다.
- 효용계산단계에는q=g인행N4rawexact/g0baselineexact/q0baselineexact와전체보완항등식assert가있다. 이단계는아직critic이실행하지않았으므로actual모든행통과주장이아니다.
- 새source/reg/audit/result v2 이름및parent_source_receiptSHA가등록됐고원parent전예측배열재사용/새fit0을유지한다. local경로는v1과같으므로v1효용산출0이라는부모상태를확인하고기존파일이있다면recipe불일치시덮어쓰지않는다.

별도JSON검사11항목이모두PASS다:현재v2sourceSHA,canonicalrecipe와감사SHA,부모recipe/registration/source receiptSHA,전체foldorderedID·원dataSHA,등록460완전날/finiteflag,감사비교셀149,040 및fit0. 비교셀149,040은460×(1+…+23+24+24)와맞는다. 이숫자는새독립표본수나원모델예측인과성검사수가아니다.

감사파일에는all23 cutoff/label거부/unchanged parent가있지만그파일존재만으로세션종료를추측하지않는다. 부모의실제exit확인과이정합receipt를함께사용해야한다. 새q의현재·과거입력감사는원모델전범위예측인과검사를대체하지않는다.

N4최종FAIL은변경하지않는다. N5는같은정답사후7번째·한PFN문맥예비가설이며첫계산결과를보고q회복/threshold를수정하지않고등록alpha7절차를그대로따른다. 채택/제출0이다.
