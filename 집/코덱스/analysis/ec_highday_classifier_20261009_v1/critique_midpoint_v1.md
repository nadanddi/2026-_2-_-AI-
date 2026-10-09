# 高EC일 분류 독립 중간 비평 v1

검토·독립 실행: recipe_v1.py/runtime_v1.py/run_v1.py/predict_v1.py/final_train_v1.py 및 사전등록/보강을 검토했다. critic_prepare_recheck_v1.py는73특징×8640행630,720셀의 RAWprefix·DP1·낮밤대비를 원 공개 입력에서 별도 구현으로 재계산했다(maxgap7.11e-15). 360일의OOF일평균라벨 high26과 모든27fold의복합farm/day query·purge·trainindex를 독립 검산해통과했다. 공식extra40y load0. 계획70열은 산술오기이며 고정코드73열로 사전보강된 것은 타당하다.

첫2fold 중간결과는 critic_score_recheck_v1.py로 독립 검산했다. 1632 OOF행/68일,24 metricgroups 각각 .5/.2 기준 pairwiseROC-AUC·threshold tie AP·Brier·logloss·precision/recall/F2/혼동행렬·농장층화bootstrap가일치했다. foldprior는각train양성일비율,ensemble은고정3seed평균이며 row/query중복은없다.

중간판정: 치명적인자료범위/누수/산술오류는발견되지않았고 동일고정코드로나머지CV를진행할수있다. 첫68일양성은1일뿐이다. h15 ensemble ROC .9851/AP .5/recall1은이1일에의존하며 안정적탐지능력의근거가아니다. .5에서TP1/FP1, .2에서TP1/FP6으로threshold를내리면이득없이오탐이늘었다. priorBrier .0197053→ensemble .0191120,bootstrap p_worse .49305/CI[-.02655,.02439]로아직분류유용성지원조건이충족된것도아니다. logit보다좋은지표도prior주기준을대체하지말것.

확인된보강: 누락prefix/새농장/중복row를assert로거부하고 빈열은imputerkeep_empty_features로유지, 단일class train은각foldassert, 한classROC는NA, 예측0분모precision은0으로고정했다. day와정답은feature73열에없고 최종400target는FULL_CVscore기록후만읽는다. 각시각은일1표본으로집계한다.

최종에유지할한계: 새FRESH7=새fold배치이며새자료아님; 일24행가중1/24은같은날독립24샘플을만들지않음; h23성공을조기예측으로확장금지;보정전probscore를검증된실제확률로표현금지. predict_v1은manifest특징/threshold를확인하지만실행시code/modelsha를직접강제하지않으므로 최종재현검사에서manifest와현재파일sha의일치를별도로확인할것. 산술검사와code-source검토는실제fit전체를독립재학습한것이아니며 최종3모델재로딩/독립probe/미래·타farm변조가필요하다.
