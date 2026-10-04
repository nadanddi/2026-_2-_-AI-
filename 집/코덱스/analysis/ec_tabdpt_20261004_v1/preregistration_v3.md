# family20 最後 실행 전 감사 해시 보완 · 2026-10-04

사전v1/v2 및 source/preparev1/v2를보존한다. 실제소스는 **run_v3.py**, 준비파일preparation_v3.json이다. 모델·학습·문턱·채택식변경0.

첫감사JSON의수치/필수검사/status/signature검사에더해정확한파일SHA를cell metadata에저장하고재개시대조한다. 저장순서는cellCSV→첫감사JSON→첫감사SHA를포함한metadata이며어느부분에서중단돼도3artifact guard가재개적합을거부한다. 준비v3와실제실행직전manifest/hash/runtime의완전일치도유지한다.

새source/사전/prepare/정적검산을main에커밋한뒤최초모델fit을실행한다. 本문서시점fit/predict0.
