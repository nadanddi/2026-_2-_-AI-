# family20 전체 verifier 준비 및 한계 v1

2026-10-04, 집/코덱스. [verify_full_v1.py](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_20261004_v1/verify_full_v1.py)를 새로 작성했다. 실제 예측 CSV·최초 감사 결과·fit audit·점수는 **읽지 않았다**. 읽은 것은 run_v4.py, 기존 family19 verify_full_v3.py, S 소스, 사전등록 문서 및 fit 전 preparation_v4의 runtime/dependency/fold 목록이다.

## 준비한 검사

- 원 R3 raw 0.8 + TabDPT raw 0.2 → 기존 causal shrink 0.5 → outer train min/max clip. LOG·KKT·inner 회귀식은 사용하지 않는다.
- 22fold×3seed의 66개 cell source/signature/manifest/runtime/CSV SHA·ordered ID와 fit audit 목록을 대조한다. aggregate oof는 66 CSV의 동일 순서·값과 대조하고 83160행·중복 부재를 확인한다.
- 기존 공개 OOF cache만 사용한 새 S.loadec의 train/query 배열·특징·목표·ID hash, baseline raw R3/PFN cache와 clip bounds를 확인한다. 원시 train_y·EL1·test loader를 호출하지 않는다.
- 최초 감사 SHA 및 고정 5 raw error, scalar error, F13/F47×hour0/6/12의 prefix·feature 기록을 확인한다. 각 day를 새 train/query 최소 day로 다시 계산해 saved day도 대조한다.
- core vector 후처리와 별도 Python math.fsum prefix scalar를 대조한다. 같은 farm-day 현재·과거 값만 사용한다.
- 5검증기×3seed의 15셀 RMSE를 NumPy와 math.fsum으로 각각 계산하고 엄격 개선 방향도 일치해야 한다.
- 기존과 같은 farm별 정렬 5일 비중첩 block, 20000회, RNG=20261003+seed의 DIAG bootstrap을 scalar/pandas 두 방법으로 대조한다.
- 일반 95% CI와 family20 조정 CI [0.00125,0.99875]를 모두 저장하며 **채택은 strict 15direction + p_worse<0.00125 + 조정 CI upper<0**이다. 95% CI로 문턱을 바꾸지 않는다.
- 기존 고EC 평균≥1인 31일/일반329일과 후기179·농장·hour0 진단을 고정했다. 진단 결과로 subgroup/가중치를 선택하지 않는다.

actual runner SHA는 **136f72ce14cf7afc078185f0e1c421052910153d6f156ebe8eed604cc3a6d2f2**, verifier SHA는 **e7891ee0dfcee43a24c12124b5075b05ee94c1775dbca7292a6f8f7323519eca**다. 기존 core/support/season/env/adapter/runtime probe hash도 코드에 고정했다. 소스가 바뀌면 기존 verifier를 수정하지 않고 새 버전을 준비한다.

## 실행한 검사

**--synthetic만 실행**했다. [verification_synthetic_v1.json](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_20261004_v1/verification_synthetic_v1.json)에 PASS를 저장했다. 고정 synthetic prefix 기대값, 실제 core shrink 함수의 AST 추출 결과, 합성 RMSE 및 개선·동점 bootstrap을 확인했다. 파일 전체는 Python 문법 검사도 통과했다. 40016은 합성 배열 값 대조 횟수이며 실제 검증 표본 수가 아니다.

fit/predict·weight 본문 읽기·실제 데이터/점수 읽기는 모두 0이다. full_verification_v1.json·full_scores_v1.csv·full_segments_v1.csv는 만들지 않았다. 이미 저장된 결과를 변경하지 않았다.

## 부모의 실행

완료 뒤 같은 승인된 읽기 권한과 고정 Python에서 다음을 실행한다. 데이터/배열·통계 검산만 수행하며 새 fit/predict는 없다.

```powershell
$env:PYTHONPATH = ""
& "C:\Users\aozks\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 -B "C:\work\farmai\집\코덱스\analysis\ec_tabdpt_20261004_v1\verify_full_v1.py" --verify
```

preregistration_v6.md, 모든 cell/최초 감사/집계/fit audit가 필요하다. 결과 세 파일이 하나라도 있으면 덮어쓰지 않고 중단한다. 검산을 통과한 경우 채택 여부가 기각이더라도 새 scores/segments와 검산 JSON을 exclusive 모드로 저장한다. default 실행은 비활성 명세 출력이다.

## 남은 한계

합성 PASS는 실제 66셀 검산 PASS가 아니다. 전체 manifest·CSV·runtime 구조는 완료 후 부모 실행에서 처음 대조된다. saved 최초 감사를 검증하지만 TabDPT 예측 자체를 다시 실행하지 않는다. 같은 설정의 fresh-fit 반복성은 부모의 최초 감사 기록에 의존한다.

반복 사용한 공개 검증이며 untouched holdout 결과가 아니다. season의 source/입력/manifest hash 재현을 확인하지만 모든 fold의 season 인과성을 별도로 변조해 다시 검증하지 않는다. weight receipt와 signature SHA를 확인하며 checkpoint 본문을 다시 읽지 않는다. 사전등록의 선행 시점은 부모의 Git 등록 증거에 의존한다(현재 runner signature에는 prereg 문서 hash가 없다).

NumPy non-writable 경고를 이유로 모델/입력/문턱을 바꾸지 않았다. verifier는 fresh S.loadec 배열 hash를 기록된 train/query hash와 대조한다. 이 검사는 transient 입력 메모리 mutation의 직접 관찰 기록은 아니다.

