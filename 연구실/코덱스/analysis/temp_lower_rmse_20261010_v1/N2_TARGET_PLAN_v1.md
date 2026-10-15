# N2 대상 CODEX 및 전체 혼합 검증 계약

LOFO gate는48농장중41(85.4167%)의두비교기준strictMSE개선과macro조건으로통과했다. 이것은대상효과아니다. 통과후공식other48농장만으로동일Ridge100/30열/전처리/손실weight를fit하고후보한개를고정한다.

- 추가특징하나=n2_cumulative. 같은농장/같은날h0=0; h1..23현재와직전14input으로trainedprior Δ추론. 현재/직전in_temp 또는시간연속이없으면ΔNaN,누적은첫NaN이후그날끝까지NaN(skipna=False),다음날재시작. h0상태를정답anchor로해석하지않음. target의정답으로prior fit/캘리브레이션0.
- 원89열/19물리열·CODEX Ridge·LGB220/.035/leaves12/minchild100/lambda15 seed9109 유지,잔차input에1열만추가. 원BASE/PFN/가중치/계절/혼합 변경0.
- 원pairedbaseline은10-09 새환경fold-local CODEX12fold;source/runtime/data/rowID/order/weight/seasonSHA接합·첫foldbaseline재fitexact프로브후reuse. 타농장prior전처리는targetfold마다재fit할필요없다(대상F13/F47와F32전부prior에서제외).
- 원입력별 feature MASK/future-globalT/otherfarm/shuffle/outer訓練외NaN 동등검사부터. 모든기존89+physical19값유지assert,1열만추가. priorScaler/Imputer source stats와훈련키/meta SHA보존.
- 현재온도대상후보4개(DROP5/MED3/IRR4/N2) 기준alpha.025/4=.00625,중앙98.75%CI(.00625,.99375),20k farm×floor(day/5) bootstrap seed20261009. 구성원DIAG10/EXT10/EXT12전부RMSE改善+p/CI통과일때만원W40G-S전체혼합 검증으로진입. 구성원실패를전체효과부재증명으로해석하지않음.
- full진입시후보와동일한새환경/訓練foldMASK/season/weights를쓰는baseline부터새학습. 온도기준모든3新BASEseed×3validator같은방향+DIAG p/CI+새PFN문맥그룹2개 동일방향을확인하고새layout확인1회.具体seed/context/layout/code/pretrainedSHA 및CPU/querybatch 계약은full fit전에別도등록. compatible아니라확인된옛W30G/GPUcache로전체확증을대체하지않음. 전체진입여부를아는것은후보탐색단계로표기.
- 새제출/확정구성/공식평가출력0. first모델후중간비평,마지막독립산술/최종비평. 사후농장/온도게이트·계수튜닝0.
