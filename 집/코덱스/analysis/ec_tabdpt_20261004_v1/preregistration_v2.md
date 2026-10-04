# family20 실행 전 guard 보완 · 2026-10-04

사전v1과run_v1/준비v1을보존하고 실제실행은 **run_v2.py**로한다. 첫modelfit전비평v15에서확인한경계를보완한다. 모델/입력/seed/context/ensemble/batch/CPU/문턱/채택식은전혀변경하지않는다.

1. core/season/env/support/adapter/probe/run hash와현재train_X/공개cache hash를모든manifest에넣는다. 원66R3metadata의train_X/core hash·5runtime을현재와대조한다. 현재준비v2 JSON과fit직전전체preflight가정확히같아야하며처음적합에도적용한다.
2. 첫감사재개는signature뿐아니라status PASS·raw/final문턱·반복/단독/역순/타query/fresh5필수검사·6prefix/6인과변조·scalar값을검사한다. 각행키·기초raw부품·범위·candidate계산도재검산한다. 부분파일은보존하고멈춘다.
3. 특징인과성대조는동일NaN pattern을허용하되다른NaN/Inf는거부한다. 정상결측입력자체를오류로판정하지않고유한값은같은1e−12문턱으로비교한다.

prepare_v2는새fit0이며, 소스/문서/준비/정적감사를main에커밋한뒤실행한다. fit전내용에차이가있으면자동재학습하거나결과로문턱을수정하지않는다.
