from pathlib import Path
import json,re,hashlib,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
assert (H/'최종혹독비평_v1.md').exists(),'Wait for independent final review'
assert (H/'순차실험_결과_v2.md').exists()
result=json.loads((H/'stage34_results_v2/completion.json').read_text(encoding='utf-8'))
assert result['status']=='SCREEN_REJECT' and result['reasons']==['101:high:protection_failed']
L=ROOT/'연구실/코덱스/local'/H.name
assert not (L/'stage1_v2/worker.lock').exists() and not (L/'risk_stage2_v1/worker.lock').exists()
live='''# 순차 실행 최종 인계 v2

2026-10-06 연구실 코덱스. 원LIVE_STATUS.md는실행중시점기록이고현재는이파일을읽는다.

- 단계1 네학습조건/1152행진단완료·독립32캐시/16day/8pair및ET/LGB0시분해PASS.
- 위험12fit 및제한보정/부분검증10152행완료. SCREEN_REJECT, seed101高보호실패. 내두workerlock제거/정상종료. 집worker상태/제어는이번범위밖.
- 부분141일全pass1/일반133高8,pass2/A/B/현재EC14효용미검증. 전체80nested결과未到着,기존snapshot만활용.
- 이득99.94~100.50%가선택F47_161에집중,이를빼면나머지140일두seed악화. F13_112미탐/F13_98部分query없음.
- 최신:순차실험_결과_v2.md,최종혹독비평_v1.md,보완기록_v1.md. 기존후보/제출물변경0.
- 다음:전체nestedとpass2공개검증지원/현재EC14정합성먼저확보,미탐및과소예측피해보호를새사전등록으로검증. 동일스냅샷문턱재탐색0.
'''.replace('と','와')
with (H/'LIVE_STATUS_v2.md').open('x',encoding='utf-8') as f:f.write(live)
cat=ROOT/'공용/데이터_단서_카탈로그.md';text=cat.read_text(encoding='utf-8');n=max(map(int,re.findall(r'\|\s*6\.(\d+)\s*\|',text)))+1
tag='EC오차해결순차실험';assert tag not in text
entry1=f'| 6.{n} | **{tag} 1단계:같은학습집합에서도성공/실패차이유지,ET/LGB일부입력반응직접분해** (10-06연구실코덱스) | 원fold0/1/8대상±1배제3집합및교집합193일,query특징해시동일.共통3seed예측F47_160 .738437/161 1.040214·RMSE.052616/.395573,F13_98 .406748/112 .675217·.023804/.279301.4조건成功2날RMSE≤.1/실패2날>.1,F13심각bias≥.2는2/4만.原48h再現R3≤2.44e-15/PFN5.424e-6(事前tol1e-5),32cache/1152행独立PASS.共통seed7/0hET差F47+.543243=CO2+.275892/난방+.250063등,F13+.264322=난방+.158826/season+.102136등;물리因果/PFN원인/24h전체기여미식별.LGB가F13에서ET보다높은반례 | 연구실/코덱스/analysis/{H.name}/순차실험_결과_v2.md·stage1_critic_v2.md |\n'
entry2=f'| 6.{n+1} | **{tag} 2~4단계:실제nested위험+제한하향보정선행기각·이득선택1날집중** | main e2ec7e0/c0c8f74 fit前고정.12LR/50feature/일weight1·보정cap.1/r>.8·환기prefixzero≥.8팬mean<10.外141日全P1(一般133高8)/10152행.一般RMSE7/101/2024−1.0976/−.5389/−.7962%,高101+.0009482%(他0)→SCREEN_REJECT.高F13_177_00実.979/予.894920→.892560/SSE+.0004024,日EC1.005125.57seed행=22고유時間행/4日,net이득99.94~100.50%가F47_161한날·除外140日두seed악화,F13_112修改0/98外query부재.独立risk再構成4.44e-16/gradient7.71e-8·전수PASS,pass2/A/B/전체80/현재EC14미검증.혹독평가/범위·경계·출처보완反映,계수튜닝/채택/제출0 | 同폴더최종혹독비평_v1.md·보완기록_v1.md·verify_risk_result_v1*.json |\n'
cat.write_text(text+'\n'+entry1+entry2,encoding='utf-8')
notice=f'> **2026-10-06 · 연구실 · 코덱스 순차 실행/최종 혹독평가 완료:** 같은학습4조건진단PASS,공통F47成功/실패RMSE.0526/.3956·F13.0238/.2793;0hET CO2/난방·계절反応범위만확인. 실제nested위험12fit/부분141日全P1 선별 SCREEN_REJECT(101高보호실패),一般−.539~−1.098%도이득선택F47_161한날99.94~100.50%집중.独立학습/10152행전수감사及최종비평완료·나머지140日두seed악화/未탐F13_112. pass2/A/B/전체80/현재EC14효용未검증;현재9회차구성변경/새제출0.보완기록/결과v2·카탈로그6.{n}/{n+1},own {H.name}/LIVE_STATUS_v2.md.다음전체nested·pass2공개지원·EC14기준정합성확보부터;같은스냅샷문턱튜닝0.\n\n'
hp=ROOT/'공용/HANDOFF.md';hp.write_text(notice+hp.read_text(encoding='utf-8'),encoding='utf-8')
diary=ROOT/'연구실/코덱스/작업일지/2026-10-06.md';log=diary.read_text(encoding='utf-8');assert '## 세션 4 — EC 오차 해결 순차 실험' not in log
log+='''

## 세션 4 — EC 오차 해결 순차 실험과 최종 혹독한 비평

## 한 줄
동일학습 비교로 fold혼동을 줄이고 입력반응 일부를 직접추적했으나, 실제nested 위험보정은 고EC보호실패와 이득편중으로기각했다.

## 한 것
- main6f8fead 단계1前등록,4조건동일query96h/R33seed/PFN4context. 원replay포함고전45fit/PFN17fit,설명ET/LGB2fit. 내worker正常종료/lock제거;집worker제어0.
- 실제nested snapshot만선택추출·manifest/16innerjob/112cache/12outer-tr.csv+query전수감사. 全80완료본없음.
- main e2ec7e0/c0c8f74 라벨/50특징/가중L2LR/조건/cap/guard 고정후12fit及제한보정10152행.
- 独立혹독비평user指示에따라최종평가재개·fresh전수감사. 처음reviewer사용량한도중단뒤22시재개,root도별도작성verifier재생. 최종평가/수치PASSと후보REJECT구분.

## 결과
| 실험 | 수치 | 판정 | 근거 |
|---|---|---|---|
| 동일학습4조건 | 공통F47_160/161 RMSE.052616/.395573, F13_98/112 .023804/.279301 | 진단완료/PFN物理원인未확정 | stage1_summary_v1·stage1_critic_v2.md |
| nested 위험+제한보정 | 一般3seed−1.098/−.539/−.796%,高101+.0009482% | SCREEN_REJECT | stage34_results_v2/completion.json |
| 이득편중민감도 | 순이득99.94~100.50%선택F47_161·제외140日두seed악화 | 사후진단/判定불변 | risk_diagnosis_scope_v2.json |

## 다른 곳에 알릴 것
- 外141日は全P1/一般133高8,pass2지원0.모든DIAG/A/B/currentEC14改善判定아님. 최신9회차EC14 .1384구성변경0.
- 57수정행はseed출현수/22고유시간행·4日. 高F13_177_00은日平均1.005125,0h정답.979·予測.89492→.89256으로손해.
- NaNcap/診断等号·50열whitelist·coverage/미탐/원기록출처를新버전보완. 기존결과/원코드保존·追加문턱탐색/원y/test/EL1/lock정답/제출0.

## 다음
- 全nestedreceipt와pass2공개검증지원/현재EC14기준정합성을 먼저확보.高/과소예측피해및LGB도높은대표未탐을새사전등록으로같이검증. family25기각原기록도먼저확인.
- 두AI작업종료후사용자가sync_end1회실행.

## 파일
- analysis/ec_resolution_sequence_20261006_v1/순차실험_결과_v2.md·최종혹독비평_v1.md·보완기록_v1.md·LIVE_STATUS_v2.md 및전수verifyreceipt. local동명폴더캐시/선택nested/fit.json/rowCSV는Drive대상.
'''.replace('と','와').replace('は','는').replace('日','일')
log+=f'\n- 새카탈로그6.{n}/{n+1}.\n';diary.write_text(log,encoding='utf-8')
print('RECORDED',n,n+1)
