from pathlib import Path
import json,re
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
assert (H/'최종혹독비평_v1.md').exists(),'Wait for final independent review'
assert (H/'순차실험_결과_v2.md').exists()
r=json.loads((H/'stage34_results_v2/completion.json').read_text(encoding='utf-8'))
assert r['status']=='SCREEN_REJECT' and r['reasons']==['101:high:protection_failed']
L=ROOT/'연구실/코덱스/local'/H.name
assert not (L/'stage1_v2/worker.lock').exists() and not (L/'risk_stage2_v1/worker.lock').exists()
live='''# 순차 실행 최종 인계 v2

2026-10-06 연구실 코덱스. 원LIVE_STATUS.md는 실행 중 시점의 기록이다. 현재는 이 파일을 읽는다.

- 단계1 네 학습 조건·1152행 진단 완료. 독립32캐시/16일/8쌍 및 ET/LGB 0시 분해 검산PASS.
- 위험모델12개와 제한보정·부분검증10152행 완료. seed101 고EC보호실패로 SCREEN_REJECT. 내 두 worker는 정상종료했고 lock이 제거됐다. 집 worker는 이번에 제어하지 않았다.
- 부분141일은 모두pass1, 일반133/고EC8. pass2·A/B·현재EC14의 효용은 미검증. 전체80nested 결과는 연구실에 도착하지 않았다.
- 이득99.94~100.50%가 선택했던F47_161에 집중됐고, 이를 빼면 나머지140일에서 두시드가 악화했다. F13_112는 미탐, F13_98는 부분외부검증에 없다.
- 최신결과: 순차실험_결과_v2.md, 최종혹독비평_v1.md, 보완기록_v1.md. 기존채택구성·제출물 변경0.
- 다음: 전체nested와 pass2 공개검증 지원, 현재EC14 기준정합성을 먼저 확보한다. 미탐·과소예측피해 보호는 새사전등록으로 검증한다. 같은스냅샷의 문턱 재탐색은 하지 않는다.
'''
with (H/'LIVE_STATUS_v2.md').open('x',encoding='utf-8') as f:f.write(live)
cat=ROOT/'공용/데이터_단서_카탈로그.md';t=cat.read_text(encoding='utf-8');n=max(map(int,re.findall(r'\|\s*6\.(\d+)\s*\|',t)))+1
tag='EC 오차 해결 순차실험';assert tag not in t
a=f'| 6.{n} | **{tag}: 같은학습집합에서도 성공/실패 차이 유지** (10-06 연구실코덱스) | 대상날±1 제외한 원fold0/1/8 세집합과 공통193일집합, query특징 해시동일. 공통3seed 예측 F47_160 .738437/161 1.040214·RMSE .052616/.395573, F13_98 .406748/112 .675217·.023804/.279301. 네조건 성공2날RMSE≤.1/실패2날>.1, F13심각bias≥.2는2/4만. 원48h재현R3≤2.44e-15/PFN5.424e-6(사전tol1e-5),32캐시/1152행 독립PASS. 공통seed7/0h ET차 F47+.543243 중 CO2+.275892/난방+.250063, F13+.264322 중 난방+.158826/계절+.102136. 물리인과·PFN원인·24h전체기여 미식별. LGB도높은F13반례 | 연구실/코덱스/analysis/{H.name}/순차실험_결과_v2.md·stage1_critic_v2.md |\n'
b=f'| 6.{n+1} | **{tag}: 실제nested 위험·제한보정 선행기각, 이득선택1날집중** | main e2ec7e0/c0c8f74 학습전고정.12LR/50특징/하루가중치1, 보정cap.1/r>.8/환기prefixzero≥.8·팬mean<10. 외부141일 모두pass1(일반133/고EC8)/10152행. 일반RMSE7/101/2024 −1.0976/−.5389/−.7962%, 고EC101+.0009482%(다른시드0)→SCREEN_REJECT. F13_177_00 실제.979/예측.894920→.892560/SSE+.0004024, 하루EC1.005125.57seed행=22고유시간행/4날. 순이득99.94~100.50%가F47_161한날이며, 제외140일두seed악화. F13_112수정0/98외부검증부재. 위험재구성4.44e-16/gradient7.71e-8·독립전수PASS. pass2/A/B/전체80/현재EC14 미검증. 최종혹독평가·경계/범위/출처보완, 계수재튜닝·채택·제출0 | 같은폴더 최종혹독비평_v1.md·보완기록_v1.md·verify_risk_result_v1*.json |\n'
cat.write_text(t+'\n'+a+b,encoding='utf-8')
notice=f'> **2026-10-06 · 연구실 · 코덱스 순차실험·최종혹독평가 완료:** 같은학습4조건 진단PASS, 공통F47 성공/실패RMSE .0526/.3956·F13 .0238/.2793. 0시ET의CO2/난방·계절반응 범위만 확인. 실제nested 위험12fit/부분141일 모두pass1에서 SCREEN_REJECT(101고EC보호실패). 일반−.539~−1.098%이나 이득99.94~100.50%가선택F47_161한날에집중. 독립학습·10152행전수감사와최종비평완료; 나머지140일두시드악화/F13_112미탐. pass2/A/B/전체80/현재EC14효용미검증, 현9회차구성변경·새제출0. 결과v2/보완기록·카탈로그6.{n}/{n+1}, own {H.name}/LIVE_STATUS_v2.md. 다음전체nested·pass2공개지원·EC14기준정합성확보부터. 같은스냅샷문턱튜닝0.\n\n'
hp=ROOT/'공용/HANDOFF.md';hp.write_text(notice+hp.read_text(encoding='utf-8'),encoding='utf-8')
p=ROOT/'연구실/코덱스/작업일지/2026-10-06.md';d=p.read_text(encoding='utf-8');assert '## 세션 4 — EC 오차 해결 순차실험' not in d
entry=f'''

## 세션 4 — EC 오차 해결 순차실험과 최종혹독비평

### 한 줄
동일학습 비교와 입력반응 추적은 진전됐지만, 실제nested 위험보정은 고EC보호실패와 이득편중으로 기각했다.

### 한 것
- main6f8fead 단계1사전등록, 네조건의같은96query시간/R3세시드/PFN네문맥. 재현포함고전45fit/PFN17fit와 설명ET/LGB2fit. 내worker 정상종료/lock제거, 집worker제어0.
- snapshot의완성16inner문맥·112구성원캐시/12outer-tr OOF만 선택추출·전수감사. 전체80완료본은연구실에없음.
- main e2ec7e0/c0c8f74의50특징/가중L2LR/라벨/조건/cap/guard 사전고정후12fit·보정10152행.
- 사용자지시로 최종혹독평가와fresh감사. 비평가사용량제한중단뒤22시재개; root도별도작성검산코드를실행. 계산감사PASS와후보REJECT를구분했다.

### 결과
| 실험 | 수치 | 판정 | 근거 |
|---|---|---|---|
| 동일학습4조건 | 공통F47_160/161 RMSE.052616/.395573, F13_98/112 .023804/.279301 | 진단완료·물리인과미확정 | stage1_summary_v1·stage1_critic_v2.md |
| nested위험·제한보정 | 일반3seed−1.098/−.539/−.796%, 고EC101+.0009482% | SCREEN_REJECT | stage34_results_v2/completion.json |
| 이득편중민감도 | 순이득99.94~100.50%가F47_161, 제외140일두seed악화 | 사후진단·판정불변 | risk_diagnosis_scope_v2.json |

### 다른 곳에 알릴 것
- 외부141일모두pass1/일반133고EC8. pass2·A/B·현재EC14효용미검증이며 최신9회차EC14 .1384구성변경0.
- 57수정은seed출현수/22고유시간행·4날. 고EC F13_177_00의하루EC1.005125, 0시정답.979·예측.89492→.89256으로손해.
- NaNcap/진단등호·명시50열·coverage/미탐/원기록출처를새버전보완. 기존결과보존·추가문턱탐색/원y/test/EL1/잠금정답/제출0. 카탈로그6.{n}/{n+1}.

### 다음
- 전체nested receipt와 pass2공개검증지원/현재EC14정합성부터확보. 고EC·과소예측피해및LGB도높은미탐을새사전등록으로함께검증. family25기각원기록확인.
- 두AI작업이끝난뒤사용자가sync_end1회실행.

### 파일
- analysis/{H.name}/순차실험_결과_v2.md·최종혹독비평_v1.md·보완기록_v1.md·LIVE_STATUS_v2.md 및verifyreceipt.
- local동명폴더의캐시/선택nested/fit.json/rowCSV는Drive동기화대상. 다른AI변경은건드리지않았다.
'''
p.write_text(d+entry,encoding='utf-8');print('RECORDED',n,n+1)
