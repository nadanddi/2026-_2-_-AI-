from pathlib import Path
import json,re,csv
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
assert (H/'최종혹독비평_v1.md').exists() and (H/'학습사례_추적보고서_v2.md').exists()
r=json.loads((H/'results_v4/completion.json').read_text(encoding='utf-8'));assert r['fits']==6
assert not (ROOT/'연구실/코덱스/local'/H.name/'worker.lock').exists()
state=dict(execution='COMPLETE_ET_SUPPORT_TRACE',independent_review='COMPLETE',new_ET_fits=6,trees=3600,nodes=34689622,query_rows=456,baseline='기존 실제 계절v2',adoption=False,physical_causality='UNIDENTIFIED',PFN_learning='UNTRACED',current_EC14_efficacy='UNTESTED',report='학습사례_추적보고서_v2.md',critic='최종혹독비평_v1.md',preregistration='78da2e5')
with (H/'완료상태_v1.json').open('x',encoding='utf-8') as f:json.dump(state,f,ensure_ascii=False,indent=2)
cat=ROOT/'공용/데이터_단서_카탈로그.md';t=cat.read_text(encoding='utf-8');n=max(map(int,re.findall(r'\|\s*6\.(\d+)\s*\|',t)))+1
tag='실제 ET 학습사례 역추적';assert tag not in t
e1=f'| 6.{n} | **{tag}: F47 실패입력이 높은EC 학습지원으로 이동한 계산경로 확인** (10-07연구실코덱스) | 6ET사전78da2e5, 원캐시차≤2.22e-16/가중정답합차≤5.55e-15. 원fold1seed7 F47_161_00 실제.663/ET1.333885, 고EC학습날가중치77.83%(고EC시간행59.67%). 학습F47_139 28%/선택EC1.632, F47_23113.5%/1.829. 공통동일ET7의160→161 고EC날가중치2→72%·예측.69099→1.23423. 첫분기난방h0 331/600의자식평균.606/1.018,CO2h0 104의.917/1.331. weightShapley CO2/난방→139일weight+21.25/+18.22%p,정답내적이원기여+.275892/+.250063재현. 원161 평활24h ET1.411361/실제.646, 최종편향+.497034중ET+.367373(한seed signed편향73.9%,RMSE비중아님). 독립3600tree/34689622node/456query fullpath·지원·permutation 전수PASS; 물리원인/훈련날삭제효과미확정 | 연구실/코덱스/analysis/{H.name}/학습사례_추적보고서_v2.md·최종혹독비평_v1.md |\n'
e2=f'| 6.{n+1} | **{tag}: F13 오류는 극단고EC 혼입보다 중간수준지원의 잘못된 적용** | 원fold0seed7 F13_112_00 실제.365/ET.681927, 고EC학습날가중치.83%뿐. F47_112지원32.33%(선택EC.736),F47_11918.17%(.837),F13_1307.83%(.797). 공통ET7 F13_98→112 자기온실지원89.5→40.5%,예측.407612→.671933;112最大지원F47_11934.33%. FULL38에온실식별자없고타온실도지원선택됨. 다만원성공F47_160도F13지원46.33%로타온실혼입자체를원인으로단정불가. 사례14개38입력/정답/지원및실제최종편향3건독립PASS. 새농장특징/분리학습/삭제재학습/최신EC14효용미검증·모델채택/제출0 | 같은보고서·examples_v3.csv·composition_v1.json·verify_trace_link_v2*.json |\n'
cat.write_text(t+'\n'+e1+e2,encoding='utf-8')
notice=f'> **2026-10-07 · 연구실 · 코덱스 실제ET 학습사례추적 완료:** 입력→실제잎지원→높은/중간학습EC→ET편향→최종EC편향 연결을 확인. 원F47_161seed7의0h 고EC학습날가중치77.83%,F47_13928%/EC1.632·23113.5%/1.829. 같은공통모델160/161 고EC날지원2/72%,난방·CO2 h0분기/weightShapley로학습날선택변화재현. 원161 하루편향+.497중ET+.367(73.9%,seed7 signed편향만). F13_112는고EC지원.83%이나F47_11232.33%/EC.736 등중간수준을실제.365에적용. 6ET/3600tree/34689622node/456query fullpath·지원·가중합·비평PASS. 물리EC원인/삭제재학습/전체61·268일단일원인/현재EC14효용미검증·채택/제출0. 결과v2/최종혹독비평/카탈로그6.{n}/{n+1}, own {H.name}/완료상태_v1.json. 다음은확인된지원혼동을새고정설계로검증.\n\n'
hp=ROOT/'공용/HANDOFF.md';hp.write_text(notice+hp.read_text(encoding='utf-8'),encoding='utf-8')
diary=ROOT/'연구실/코덱스/작업일지/2026-10-07.md';d=diary.read_text(encoding='utf-8') if diary.exists() else '# 2026-10-07 · 연구실 · 코덱스\n'
assert '## 세션 1 — 실제 ET 학습사례 역추적' not in d
entry=f'''

## 세션 1 — 실제 ET 학습사례 역추적

### 한 줄
이전입력대비진단을 실제트리의학습지원까지연결했다. F47은높은EC지원으로, F13은중간EC지원으로이동해과대예측됐다.

### 한 것
- 사용자 “추적해봐”. 10-06시작/10-07완료. system debugging/독립혹독비평 적용. 이전공개360일정답만읽고원y/test/EL1/잠금정답0.
- main78da2e5 사전등록. 공통3seed/추가purged1/원fold0·1seed7 총6ET를원설정대로재현. sourcev1~3은fit0준비버전,실행v4만6fit. 경로치환실패/UTF8비교오류보존후새버전보완.
- bootstrapFalse/squared_error/균등지원,실제float32경로와잎을저장. 모든학습정답의기여를1/(600*n_leaf)로재현.
- 학습지원의hour0/raw-day/고정평활day·고ECrow/day·온실·훈련날을구분,5그룹32조합 weightShapley 및부모/양자식학습통계를추적.
- 독립검토자가3600tree/34,689,622node의훈련/query/조합경로를재구성하고120순서Shapley·456query·14사례·3최종편향을검산. 새fit0,최종혹독평가및문안보완완료.

### 결과
| 추적 | 수치 | 판정 | 근거 |
|---|---|---|---|
| F47실제실패원ET7 | 161일0h고EC날지원77.83%,139일28%·선택EC1.632/231일13.5%·1.829 | 계산경로확인 | results_v4·summary_v1 |
| 동일모델통제 | 공통160/161 고EC날지원2/72%,예측.690990/1.234233 | 입력에따른지원이동확인 | weight_shapley_rows·first_split_training |
| 최종편향연결 | 원161seed7총+.497034, ET가중항+.367373(73.9%) | signed편향분해PASS | composition_v1.json |
| F13실패 | 원112실제.365/ET.681927,타온실112지원32.33%·선택EC.736 | 중간수준지원혼동확인 | support_days·examples_v3.csv |

### 다른 곳에 알릴 것
- 근접거리이웃이아닌실제모델지원으로확인했다. F47과F13오류유형은다르며모든실패를고EC날혼입으로설명하지않는다.
- 73.9%는원seed7 하루평균편향의분해다. RMSE비중·3seed전체·물리원인·삭제재학습효과가아니다. 타온실지원은성공사례에도있다.
- 이전실제계절v2기준이며최신9회차EC14구성변경/새제출/새모델채택0. 카탈로그6.{n}/{n+1}.

### 다음
- 입력에따른지원혼동의일반화범위와보호조건을새사전등록으로검증한다. 온실특징/훈련날제거가효과있다고단정하지않는다. 기존fullnested/pass2/currentEC14대조는별도확보과제다.
- 두AI작업종료후sync_end1회. fullforest NPZ약360MB는내local·Drive동기화대상,원캐시/다른AI파일변경0.

### 파일
- analysis/{H.name}/학습사례_추적보고서_v2.md·최종혹독비평_v1.md·보완기록_v2.md·완료상태_v1.json·results_v4/summary_v1/verify receipt.
- local동명폴더에6개fullforest/훈련-query 배열/정답/leaf/가중치/메타데이터보존.
'''
diary.write_text(d+entry,encoding='utf-8');print('RECORDED',n,n+1)
