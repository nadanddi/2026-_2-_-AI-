# 중간 코드 독립 비평

2026-10-09. PLAN_v2/runtime_v1/audit_v1/run_v1 및 공식패키지 split_mask·diag_folds 읽기만. 새fit0/외부자료0.

## 수정 필요: future 교란 절단의 정의

audit_v1은모든평가일의hour>5 또는17을변조하고모든평가일hour<=cut을검사한다. 이때이전평가일의오후는다음평가일오전의허용과거다. BASE연속EWMA가그허용과거교란에반응해도FAIL로나오므로인과성위반과구별되지않는다. **전체시각T=day*24+hour 한점을정하여hit.t>T와take.t<=T로검사하는새audit_v2가필요하다.** 기존실행에서FAIL이발생해도그것만으로모델미래누수를주장하면안된다. MED3의당일reset 미래검사에는현재hour기준도타당하지만원BASE검사에는불충분하다. root에게즉시전달했다.

## 나머지 코드 검토

- runtime은공식train_y에서row_id/sub_temp만읽고sub_ec는NaN으로생성한다. 원TM/V의ORIG는이교체loader를참조해EC숫자정답을프로젝트재로드하지않는다. 원raw센서의다른농장데이터는공식입력범위다.
- CODEX frame은lab.row_id순서로reindex후RangeIndex를복구하고target은lab의같은행순서배열로붙인다. trm/vam은원lab의원day로split_mask±1일버퍼를계산하여특징frame과동일위치에사용한다. season변환전mask를고정하므로season day값으로split하지않는다.
- MED3는farm/day/hour정렬격자0..23 rolling3/min_periods1로현재h−2..h이고원89생성기를재사용한다. act_co2/물리19는변하지않으며drop5는해당5열만LGB에서제외한다. baseline/drop/med의물리Ridge는같은입력·target·weights로별도fit해동일비교조건이다. 세조건이지새후보3개가아니다.
- 매fold훈련외rawinputNaN세계와originalworld의CODEX/MED3훈련89열동등assert로검증·purge입력경로를통제한다. 실제assert통과전소스상검사존재와실행결과를구분한다. fullBASE는이번fit에사용하지않으며FULL BASE는EXTcold정의용으로만계산한다.
- weights는tr.row_id로rawtrain_X를제한해TF._train_frame/noisyrank/coldEWMA를계산한다. 최종train label을weights가참조하지않고구간gap처리가기존TF와같다는새정의를기록한다. season은tr farm/day만raw선택→vectors→mapping으로정규화/anchor/PAV를fit하고va는day보간한다. season_frames에는train/querykey불교집합assert와selectedkey==trainkey증거를추가하면출처감사가더명확하다.
- bootstrap은같은농장×floor(day/5)블록의SSE와count를20k재표집해RMSE차를계산한다. seed20261009/중앙97.5%CI .0125,.9875/p<.0125가등록되어통계정의가명확하다. 후보별동일draw를사용해진단을비교한다. trainseed9109독립반복이라고주장하지않는다.
- 신규local캐시는fn존재시metadata recipe_sha를검사하고재사용한다. external cache는사용하지않으나NPZ해시·rowID/queryID대조가현재없어실수덮어쓰기/파일교환을잡지못한다. 첫fold산출부터NPZ SHA를receipt에기록하고최종검산에서query_ids및shape를확인하도록권고한다. 최종new_fit=count*3는현재프로세스신규fit수라첫fold재사용후전체누적fit수와구분한다.

품질future검사수정과실제첫audit증거를확인한뒤첫fold진행이타당하다. 원자료/원소스SHA와5열변경영역/마스크/격리/행셔플/가중치검사결과의실제통과를받기전코드존재만으로규정PASS를주장하지않는다.
