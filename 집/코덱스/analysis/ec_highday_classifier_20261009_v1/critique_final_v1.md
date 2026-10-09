# 고EC일 분류 독립 최종 비평 v1 — 2026-10-09

판정: 사용자확정 일평균EC>=1.2의 분류기 제작·저장·재현은 완료됐고 검증 산술은 통과했다. 고EC 구분 신호는 보이지만 사전 prior대비유용성지원 조건은 실패했다. 회귀보정·기존EC후보채택·제출을 만들지 않는 결론이 타당하다. '모델완성'을 '안정적고EC탐지/실전효용확증'으로 확장해서는 안 된다.

직접 검토와 독립 실행:
- 원공개OOF/train_X 73특징×8640행630,720셀과360일타깃·27fold purge를 critic_prepare_recheck_v1.py로 독립 재계산해통과(maxgap7.11e-15). 공식extra40정답은 이 단계에서읽지않았다.
- critic_score_recheck_v1.py midpoint/final 직접실행PASS. 최종10848행/144metricgroups의pairwiseAUC·tie-threshold averageprecision·Brier·logloss·.5/.2혼동/F2와농장층화20,000bootstrap를별도산술로 재계산했다. fitreceipt/고정코드 검토이며 CV108fit 전체를독립 재학습한것은 아니다.
- critic_diagnostics_recheck_v1.py는8개farm/pass층의오류·AUC·Brier,13개reliabilitybin과9개동일ETmember/all24시간 학습·검증Brier 집계를독립 확인했다. 학습Brier는receipt의예측점수를재집계했으며raw학습확률이저장되지않아직접재계산은아니다. validation은원OOF에서직접 재계산했다.
- critic_final_model_recheck_v1.py는독립 재생성으로확인한공개73feature를192probe키에정렬해저장3joblib를직접predict했다. 확률평균과CLI결과gap8.67e-19,분류class동일;현재code/source/modelSHA모두manifest일치. 공식400라벨은FULL_CV이후평균·count정합만검사했고extra40의성능예측/재검증은0이다.
- 루트의실제추론인과성검사소스와실행JSON을확인했다(비평가가그변조검사를새로실행한것은아님). 두농장×두구간×h8/h15=8case에서미래/타농장변조·제한입력·입력순서 gap0,중복ID/새농장/누락h0거부. 실제CLI192행재생성도gap0. 이는검사한입력범위의근거이며모든가능입력에대한증명은아니다.

주분류결과(h15,threshold.5):
- DIAG전체360일(고EC26): ensemble ROC-AUC .8209351/AP .3821581/Brier .0545482,TP8/FP6/FN18/TN328. precision .5714/recall .3077. 26고EC중18일을놓치므로높은accuracy .9333으로능력을과장하면안된다.
- FRESH7 46일(고EC5):AUC .9804878/AP .8761905/Brier .0465961,TP3/FP1/FN2.
- EL1 동일46일(고EC5):AUC .9707317/AP .8600/Brier .0488375,TP4/FP1/FN1. FRESH와EL의고EC는같은5일이며각각3/5·4/5를새양성10일처럼합산할수없다.
- DIAGh15 priorBrier .0678103→ensemble .0545482로점추정은좋고모든시드/세검증기에서prior보다작다. 그러나주bootstrap p_worse .06465/.025불통과,ΔCI95[-.0320597,.0036539]의상한>=0로고정지원조건FAIL. logit보조도p .0393/CI상한 .0014089>=0이므로prior판정을구제하지못한다.

농장/구간과과적합:
- DIAGh15 F13 1차고EC10일을전부놓쳤다(TP0/FN10/FP4). F13 2차는1/3탐지·FP1이며 F47 2차는2/2탐지·FP0다. F47 2차AUC/AP1.0은고EC2일표본에의존한다. FRESH/EL의F13탐지는1/3·2/3으로구성에따라달라진다. '두농장에안정적'이라는주장은부적절하다.
- 동일member·동일24시각분포로맞춘9쌍의fold평균trainingBrier .009755~.010562 대 OOFvalidation .057125~.062278의격차가크다. 훈련성능을그대로미래에기대할수없고과적합/분포차위험을시사한다. 학습은fold평균,검증은query행평균이라가중치/분포차도남는다. 3seed학습Brier평균을ensemble학습Brier로부르면안된다.

보정과한계:
- DIAG 점수 .75~1 bin은2일/실제고EC1일뿐이고평균score .765 대양성비율 .5다. 이를정확히보정된확률이라부를수없다. '보정전고EC확률점수'로한정하며 .2민감도결과를근거로임계값을사후채택하지않는다.
- h15특징은당일15시까지관측하며목표는당일24시간일평균이다. h23성과는조기예측성과가아니다. FRESH7은새fold배치이지새자료가아니고기존라벨노출·특징탐색을해소하지않는다. 24시각을독립일로합치지않은평가는타당하다.
- 최종모델은400일9600행/high30·normal370으로학습됐다. Decimal/fsum/pandas기본/round_trip 집계모두30,정확히1.2/1e-12근처일0으로라벨경계정합PASS. 과거29기록차이를부등호/float경계로단정할근거는없으며이전모집단복원은이번범위밖이다. 360일CV는400일최종모델의직접홀드아웃성능이아니다.

개선반영/마무리: 계획73열산술정정·양class예외·빈열유지·MASK적격·엄격purge·농장층화·입력거부규칙을반영했고최종보고v2는같은member/all24시간 Brier로비교를고쳤다. 추가필수코드수정은없다. 기록은 '분류기제작·재현완료/구분신호존재/사전유용성확증실패/보정·회귀채택·제출0'으로남기는것이타당하다.
