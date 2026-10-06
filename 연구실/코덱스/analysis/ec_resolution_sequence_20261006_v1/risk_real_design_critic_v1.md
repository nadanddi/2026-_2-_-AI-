# 실자료 FULL 화이트리스트/helper 경계 독립 확인

2026-10-06. verify_risk_real_design_v1.py로risk_inputs_v1의24CSV전수에risk_core_v2.design(frame,M.FULL)을적용했다. **PASS_REAL_DESIGN_BOUNDARY_ONLY.** receipt verify_risk_real_design_v1_20261006T182641394539.json에24파일SHA와정확한base/design열목록을고정했다.

명시FULL은38열,design은50열이다. 추가12열은farm_F47,A,smooth_et/smooth_lgb/smooth_mlp/smooth_pfn,prefix_A,et_minus_lgb/mlp_minus_lgb/pfn_minus_lgb,member_range,A_minus_lgb이다. FULL외정답/sub_ec/inner_j/row_id/기타라벨은추가되지않았다. 온실정보farm_F47는명시파생피처로사용된다.

모든24파일에서기존prefix_A와시간순재계산값이일치했다. sub_ec/inner_j/row_id와다른known라벨을변경하거나추가해도50열값이변하지않았다. 입력행shuffle후원index에맞춘결과도전수동일했다. 파일별두온실의중앙cut에서미래행의FULL·모델점수/A를바꾼경우와타온실행을바꾼경우,해당온실현재·이전행의design은변하지않았다(총48file-farm cut).

범위는고정FULL과원모델점수를받은helper의경계다. raw입력을바꾼뒤season/원기본모델을재fit·재예측한인과성검사는아니며실제위험모델fit/후보선택/문턱등록을하지않았다. 지금감사한38/50열은위험프로토콜에서그대로선택하거나변경시새사전등록할수있는구체적인열목록이지자동채택된후보모델이아니다. 이전의부분nested/서로다른inner·outer모델분포/고EC보호및전체DIAG/A/B확증제약은유지된다. 단계1완료검산은실행하지않았다.
