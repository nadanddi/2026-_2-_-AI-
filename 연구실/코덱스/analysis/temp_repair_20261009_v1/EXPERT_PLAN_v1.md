# 전문가 단서 기반 유효일사 CODEX 이식 · 사전 계획

기존 두 CO₂ 후보와 독립인 입력 상호작용 후보다. 전문가_온도반영_v1.md에 CO₂ 결과 공개 전에 명시한 4열 설계를 그대로 시험한다. 전문가 설명이 성능을 보장하거나 멀칭 효과량을 주었다고 해석하지 않는다. 기존 BASE에 같은 정의가 있으므로 전체 모델의 새 물리 지식이 아니다.

- 후보 하나 IRR4: CODEX 잔차의 기존89열에 rad_eff=out_rad*(act_shade.fillna(100)/100)*(act_thermal.fillna(100)/100) 및 같은 농장·당일 EWMA halflife1/3/8 네 열 추가. Ridge19열·가중치·계절·LGB220/.035/12/100/lambda15/seed9109 고정. 원 act_shade/thermal 방향은 패키지 common.add_derived와 같은 입력 정의이며 독립으로 확인한 장치 상태라고 주장하지 않는다.
- 학습/검증 입력은 officialtest14열 MASK 세계. 실제 query마다 현재·과거만, day reset. 미래·다른 농장·행셔플/훈련외입력 MASK 동등검사 먼저 시행. 전문가 수온/멀칭/센서depth 값은 모델에 넣지 않는다.
- 기준은 바로 끝난 새환경 CODEX 기준. 같은12fold/동일row순서/훈련keys/weights/season과입력 SHA를 재사용한다. firstfold baseline 다시fit해서 raw pred exact 일치 확인 후 나머지 기준cache 사용. 원 W40G/PFN/TK cache 사용 아님. 모든 NPZ hash/recipe/queryID 전수 확인.
- 3후보 전체 탐색에 대해 IRR4 진입 기준을 alpha=.025/3=0.0083333333, 중앙98.333333% CI(quantile .0083333333/.9916666667)로 고정. 기존 두 후보 판정은 원가정에서 이미미달이므로 재채택하지 않는다. farm×floor(day/5)20k bootstrap seed20261009. DIAG10/EXT10/EXT12 전부 RMSE개선+p/CI통과 시에만 fullW40GS 새seed/context 검증 단계로 진입. 구성원 진단만으로채택금지.
- 원공식 sub_temp400일만 사용. EC정답수치/다른대회자료 사용0. seed반복을독립확인으로세지않는다. 끝나면독립 산술/최종비평, 연구보고서와재개지점 기록. 신규 제출/공식평가예측0.
