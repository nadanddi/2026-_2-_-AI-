# 도메인24 실제 준비·fit 등록 독립 점검

2026-10-07. register_domain_BLK_fit_v3.py / run_domain_BLK_raw_v3.py / DOMAIN24_BLK_fit_registration_v3.json / preparation v2 / event audit 및 feature_candidates_v5.csv를 읽었다. PowerShell로 registration의 실제91source SHA를 재계산해 불일치0, runner/preparation SHA 일치를 확인했다. 모델 fit/채점/GPU 실행 0.

## core fit-blocking 판정

새로운 핵심 누수·가설 미구현·fit 등록 누락은 발견하지 못했다. 고정 source를 유지하여 baseline ET replay3회→전체24×3seed rawfit72회를 진행할 수 있는 source·등록 근거다. 실제 replay/fit 성능 또는 mixed pipeline gate의 PASS 판정은 아니다.

## 주요 기존 지적 닫힘

- exact24 family_map, preparation SHA 및 전체 source/runner pin이 등록됐다. runner는 selected/prep/reg map 집합과 map equality, 실제 추가열을 확인한다.
- 준비 receipt는160prefix/future 및17856scalar,6960행·1187열이다. 새3series lag/rate/d2 대조가 들어 있다. 실제 source/receipt pin이 일치한다.
- event receipt는8series×7시점×32current/dynamics=1792scalar 대조 PASS를 기록한다. 독립 scalar expected의 current·lag·rate·d2·rolling 계산, h0/gap, onset/offset 및 actuator/environment NaN anchor reset, sealed_run/prefix가 source상 확인된다. 이 기록은 준비 감사와 별개이며 전체 모델 누수 검증을 대신하지 않는다.
- v3 resume는 replay matrix·현재 원baseline file SHA/seed와candidate ID/seed/member/finite1440/matrix/3audit difference/emptytraincolumns·실효imputer열/width를 재검사한다. 실제 fit에는 imputer outputnames/width/statSHA와mean1/sum1 structural aliases를 기록한다.
- model factory ET600/max_features1/minleaf1/jobs4와 등록이 맞고 predict jobs1로 바뀐다. threadpool_limits1은 BLAS thread 제한이며 ET jobs4 설정과 동일 항목이 아니다. median imputer는 reference행만fit한다.

## 등록의 제한·좁은 문서 메모

통계는24×2scope48안과기존6합산54,alpha.025/54,seed3전체개선,p ties>=0/20001,20000draws 및개별descriptive95CI로 고정됐다. 과거 전체196탐색/원검증기/최초미사용확정을 대체하지 않는 한계와 모든24원TM/P2LOO/EL1 필수 조건이 기록됐다. BLK로 나빴던 후보를 제외할 권한은 없다.

v5 CSV는 D05/D09/D11/D13/D22의 실제 event 의미로 설명을 정정했다. 단 reg.family_map은 preparation.family_map을 그대로 가져와 `difference`가 옛v4 설명일 수 있다. 보고·후속validator 등록에서는 source formula/exactcolumns/v5설명을 사용하고 prep의 stale text를 구별하면 된다. 계산 열이나 새 가설을 바꾸는 문제는 아니며 이번 rawfit을 막는 결함으로 판정하지 않는다.

원24 논리 grammar를 유지하여 alias/split-sampling 효과가 섞이며 D09/D22 VPD event series가 겹친다는 한계는 명시돼 있다. 원 submission14/GPU OOF 동일재현이 아니다. 이번 audit와 등록은 source/preparation 수준의 진단 준비이며 실제 raw모델 순서/단일batch 검사, 최종mix/shrink/clip/SG2의모든후보 인과gate, scorer등록 및 독립산술검산은 이후 확인해야 한다.
