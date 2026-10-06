# 위험helper 및 인과입력 검사 추가 독립 비평

2026-10-06. risk_core_v1.py/test_risk_core_v1.py/causal_inputs_v1.json 검토. verify_risk_core_contract_v1.py로96개유한파라미터격자와12개FORBIDDEN열거부/정답변경/미래·다른온실변경/시간순재정렬불변을독립검산했다. 위험fit0,후보선택0. test RED→GREEN기록은helper구현계약검증이며위험모델성능검증이아니다.

## 판정

유한한threshold=.8/cap=.1에대한96수치계약은PASS:보정후0≤p'≤p,변화량≤cap,gate거짓/위험점수문턱이하/기준예측이원예측이상인경우변화0을확인했다. design의명시base_columns및knownFORBIDDEN차단,sort후같은온실·하루prefix_A재계산도테스트범위에서PASS다.

**확인된경계결함:**cap=float('nan')은 cap<0 검사에서거부되지않아downward가NaN을반환한다. 독립결과상태는PASS_FINITE_PARAMETER_CONTRACT_WITH_CAP_NAN_DEFECT이며무조건PASS가아니다. 실제후보cap은유한상수로등록될예정이라현재자료누수나모델실패증거는아니지만'항상유한·범위보장'계약을제시하려면새버전에서cap유한scalar검사및해당회귀테스트가필요하다. 원helper는수정하지않았다.

## 화이트리스트와 인과범위

- FORBIDDEN은알려진12열이름을차단할뿐임의의미래/라벨파생열을차단하는화이트리스트가아니다. 다음위험프로토콜에서실제로사용할FULL/BASE/점수차/운영gate의열목록을명시고정해야한다. base_columns에아무숫자열을넣을수있는순수helper만으로누수방지를완료했다는주장금지.
- farm_F47은메타의문자열farm을명시적으로이진피처로변환해사용한다. rawfarm이FORBIDDEN이라도온실정보자체를전혀안쓴다는뜻은아니다. 실제feature선택과그사용근거를등록한다.
-현재helper는A·네평활점수·prefix_A·구성원차이·range를항상추가한다. 이점수는원모델의합법시각별출력이어야하며모델출력의causality를helper가자동으로검증하는것은아니다. 보관된prefix_A는신뢰하지않고시간순으로재계산하는점은긍정적이다.
-causal_inputs_v1.json은두온실×3cut의37개비계절입력과입력순서불변을검사하고season/model_predictions검사를false로별도표시한다. 범위표기는정확하다. 이를FULL38의전열/PFN추론/보정후출력까지causal이라고확대하지않는다.
-helper테스트의메타변경은그예제에서그룹구성을유지한일차변경/라벨값변경불변이다. day/farm그룹자체를임의로바꾸어도prefix결과가불변이어야한다는계약은아니다. farm과day/hour는표시피처제외와별개로정렬·그룹의필수정상메타다.

위경계결함수정과실제caller화이트리스트고정후후보코드를검토해야한다. 지금은일반helper와부분자료감사까지만검증됐고실제후보C/라벨문턱/risk문턱/cap/고EC보호gate 또는모델학습의정당성을검증한것은아니다.
