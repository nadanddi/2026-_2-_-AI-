# EC 현재 외기4열 추가 대조 — 실행 전 고정

입력 가용성 정정: test_X MASK에서 out_temp/out_hum/out_rad/out_wspd 모두1440/1440행 존재한다. test_X 값분포나목표는읽어학습하지않았다. 앞선 'query외기 가려짐' 설명을철회한다. 기존모델 FULL38은 외기4열을직접쓰지않으며 season학습에날씨를이용한다. 다른모델/다른레시피의weather실험은있으므로전혀미시험정보원으로표현하지않는다.

현재 season-v2의ET부분에현재행외기4열만추가(FULL38season-last뒤 W4,42열). 행ID로train_X 현재4값을일대일합치며 전일집계/날씨동일쌍flag/미래입력/다른온실query입력을추가하지않는다. 기존계절변환/RAW10현재이전/ET600leaf1/seed7/101/2024/22fold/PFN다른멤버 유지. 후보=clip(ref+.48*(새ET당일평활-기존이미평활ET),outer학습목표범위). 모델·혼합·하이퍼파라미터·날씨변환 튜닝없음.

family12: 모든15검증기×seed개선, DIAG농장층화5기록20000bootstrap p_worse<.025/12 및CI상한<0. 공개통과도EL1/추가감사전채택아님. 첫fold새학습재현0·전체행scalar후처리/분할/단독추론/현재이전허용 입력감사,RMSE/블록bootstrap독립검산. raw EC/잠금/test예측/제출물생성없음. MASK검사외test_X읽기없음. 여러날weather/직전실제둘째flag/계절표변경안과구분해판정한다.
