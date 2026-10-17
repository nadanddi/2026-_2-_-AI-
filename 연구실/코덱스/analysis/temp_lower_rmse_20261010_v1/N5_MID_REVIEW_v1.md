# N5 첫 fold 독립 중간 비평

2026-10-10 연구실 코덱스. `n5_critic_first_v1.py`를별도로실행하여정상exit0을확인했다. 원모델fit/predict0·EC수치정답0·외부자료0. critic결과는 `n5_critic_first_v1.json`에신규저장했다.

**판정: 첫fold의원입력prefix재계산·부모배열/출처·게이트·혼합검산이통과했다. 같은 `n5_run_v2.py`/registration_v2로전체12fold 계산을계속할수있다.** 첫fold 개선을보고설정을수정하거나전체기준통과로말하지않는다.

## 실제 별도 검산

- 현재N5source/부모registration·canonicalrecipe/source receiptSHA,원3CSV SHA·부모fold ID를검사했다. 첫부모N4 NPZ/receiptSHA와N5저장NPZ/recipeSHA도정합한다. query960ID순서가부모/새배열/등록/receipt에서동일하고ID중복·train교집합0이다.
- 공식train_X/test_X의row_id/in_temp만읽어농장·날·시각정렬후 **pandas expanding을사용하지않는별도scalar루프**로prefix최저를계산했다. 현재NaN이면이전유효최저를유지하고이전유효관측도없을때만NaN/q1을사용했다. 전체460날0..23시완전지원,같은날q비증가/q≤g가맞다.
- 첫960행의storedprefix_min은NaN위치까지scalar재계산과같고q/g도exact같았다. 원g는부모N4gate와exact같다. 현재원rawinput이변경되지않았고parent최종배열은그대로재사용된다.
- N5의BASELINE배열은N4baseline과raw exact동일. q=g인행은N4후보와exact동일이고q0/g0행은기준과exact같다. q0 117행(원g0 36행),q<g 223행으로예정한추가억제만있다. 이행수는새효용이득의원인이나단위독립표본수가아니다.
- 원부모BASE/CODEX/IRR/PFN멤버부터scalar fsum으로전체혼합을검산한최대차3.55e−15,출력유한값전수PASS다. 공개온도query정답은예측후산술검산에서만usecols로읽었다.
- 독립fsum RMSE: **기준 .7271330403509996 → N5 .724821062846107**. ΔMSE−.003356885225=교차항−.003554527852+보완MSE+.000197642627의항등식도확인했다. 첫fold bootstrap·최종효용검정은하지않았다.

## 진행과 해석 범위

부모가확인한audit53470 actualexit0/149,040gate비교셀과이번별도원입력재생검산을연결한다. 부모모델을새로재학습/예측한것이아니며전체pred인과성/배치독립성/PFN8/모든실제seed를검증한것도아니다. 이번검사는불변부모예측에적용한신규gate와혼합의첫공개fold다.

전체는같은원12cache·기준배열·계수·threshold·원g·등록alpha.025/7/pooled20k·RNG20261010을유지한다. 첫cache는SHA대조후재사용하고추가fit0으로나머지계산을마친다. 최종모든3validator방향·DIAG통계기준과독립최종비평이끝나기전PASS/채택/새제출0이다. FAIL이면q회복시간·문턱·농장별선택을사후붙이지않는다.
