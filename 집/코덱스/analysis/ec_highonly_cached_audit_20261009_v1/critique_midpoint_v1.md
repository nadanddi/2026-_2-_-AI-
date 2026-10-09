# 고EC전용수치회귀 캐시 독립 중간 비평 v1 — 2026-10-09

판정: 원자료/캐시정합과집계산술에치명적오류는없어고정run_v2.py로전체10fold감사를진행할수있다. 처음2fold의고EC는1일뿐이라이숫자는재현/버그점검이고전체26일성과판정이아니다.

검토/독립실행: 지적을반영한PLAN_v3/run_v2는CT1sourcepin과숫자finite검사를추가했다. critic_recheck_v1.py midpoint를직접실행하여원MX1/CT1의8640개row_id/farm/day/hour/y/fold일치,독립farm/day purge재구성·원DIAGtrainindices일치,10fold학습highcount·highconstant를재계산했다. 중간고EC1일24행의18groups/3segments에서RMSE/MAE/bias/예측·실제mean/std(ddof0),하루평균RMSE/MAE와<=.1/.2/.3일수모두math.fsum 산술과일치했다. 공식추가40정답·성능/test값은읽지않았다.

중간값: 이1일에서BASE평균RMSE .2962779, H평균 .4743157, highconstant .5435932다. 실제일평균1.2108333 대BASE1.0581152/H1.3930978이며두모델각각일평균절대오차 .1527181/.1822644다. 이는한사례의고EC수치조건부결과이며H의일반적우열이나어떤시각분류성능도말하지않는다.

학습범위: 모든DIAGfold high학습count는13~22일이며고ECquery가있는fold는13~20일인범위를구분한다. high전용fit의입력season은전체fold입력에서만들어졌고공통shrink와전체outertrainlabel범위clip이적용됐다. 그래서일반일입력/후처리까지전부삭제한순수새실험이라고부르면안된다. 원모델간접의존코드전체와cache생성당시학습/누수구조를재현한것도아니다.

남은작업: 동일26일전체10fold·3seed·BASE/H/상수,두농장×앞뒤구간·하루평균오차를같은독립checker로검산한다. 검정/채택0,새fit0/새LOO0/새최종모델0이다. 조건부회귀감사의답을실전전체회귀·고EC선별·strict일반정보삭제의성과로확장하지않는계획을유지할것.
