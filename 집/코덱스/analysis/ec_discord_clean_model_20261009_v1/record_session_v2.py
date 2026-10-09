"""Append the completed session after independent model/CV gates pass."""
from pathlib import Path
import sys,json,re
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'집/코덱스/local'/H.name
assert (H/'critique_final_v1.md').exists()
s=json.loads((H/'final_score_v1.json').read_text(encoding='utf8'))
a=json.loads((H/'final_artifact_independent_v1.json').read_text(encoding='utf8'));assert a['status']=='FINAL_ARTIFACT_INDEPENDENT_PASS'
r=json.loads((H/'root_final_cv_recheck_v1.json').read_text(encoding='utf8'));assert r['status']=='PASS'
c=json.loads((H/'final_model_causality_v2.json').read_text(encoding='utf8'));assert c['status']=='PASS'
fullcheck=json.loads((H/'full_cleaning_recheck_v1.json').read_text(encoding='utf8'));assert fullcheck['status']=='PASS'
d=json.loads((L/'dataset_full_train/cleaning_manifest_v1.json').read_text(encoding='utf8'))
G={(g['scope'],g['arm'],g['seed']):g for g in s['groups']}
f=lambda scope,arm,seed='ensemble':G[(scope,arm,seed)]['rmse']
pct=lambda scope:100*(f(scope,'CLEAN')/f(scope,'BASE')-1)
cat=ROOT/'공용/데이터_단서_카탈로그.md';old=cat.read_text(encoding='utf-8-sig');n=max(int(x) for x in re.findall(r'^\| 6\.(\d+) \|',old,re.M))+1
basepath='집/코덱스/analysis/ec_discord_clean_model_20261009_v1'
modelpath='집/코덱스/local/ec_discord_clean_model_20261009_v1/model_full_clean_v1'
table='| 검증 모집단 | 일수 | 원자료 R3 RMSE | 정제자료 R3 RMSE | 변화 |\n|---|---:|---:|---:|---:|\n'
for scope,label in [('filtered','정제조건부'),('all','원검증 전체'),('removed','조건부제외일'),('normal','일반일'),('high','고EC일'),('pass2_all','2차 원검증 전체')]:
    g=G[(scope,'BASE','ensemble')];table+=f"| {label} | {g['days']} | {f(scope,'BASE'):.9f} | {f(scope,'CLEAN'):.9f} | {pct(scope):+.3f}% |\n"
probs={v['scope']:v['p_worse'] for v in s['bootstrap']}
text=f'''\n\n## 세션 2 — 설명하기 어려운 고EC일 통삭제 데이터와 EC 모델 제작

### 한 것
- 사용자 새 지시대로 원자료를 보존하고 고EC 불일치일24시간 전체를 X/Y에서 같이 삭제한 새 자료 및 EC모델을 제작. 규칙은 이전과동일: 학습일평균EC>=1, 같은농장±1제외47특징5NN의 EC중앙보다 .5초과; 한차례선정/재귀삭제·임계변경0. 물리적이상/라벨오류 확정 아님.
- 공개360일 정제자료는12일288행제거→348일8352행. CV에는 이global목록을사용하지않고 원360에서 원DIAG10±1분할을유지해foldtrain 안에서만 선정. 검증 q의 제거는 trainreference와q정답일평균으로 결정하는 조건부채점용,훈련에는qlabel유입0. 각fold모든selector행렬/이웃/원값유지/cleanX/Y/훈련ID보관.
- 각fold의삭제후season/imputer/scaler/bounds와ET/LGB/MLP모두재fit. 새R3모델=.6ET+.3LGB+.1MLP/seed7,101,2024/prefix평균.5·현재.5/clean범위clip. PFN/SG2는이별도R3모델구성에없음,기존EC14동일구성이라고주장하지않음. 동일R3 baseline cached원구성원raw와동일검증행으로비교.
- 등록후CV모델90(각kind30)+baseline ET/LGB/MLP replay3. 기준gap ET0/LGB0/MLP3.1086244689504383e-15. CV완료score 저장뒤에만 공식9600행400일 train_y load,소비된40일은최종학습에만포함/40재검증0. 최종400일에서{d['removed_days']}일{d['removed_rows']}행통삭제→{d['after']['days']}일{d['after']['rows']}행. 최종9모델전부이clean자료학습·joblib저장.

### 결과 (카탈로그6.{n})
{table}
- 조건부p_worse={probs['filtered']},원전체p_worse={probs['all']},공동기준alpha.025/2=.0125. 표의변화는항상같은검증행에서baseline과candidate비교하며,서로다른모집단끼리의RMSE차를삭제효과로해석하지않음.
- 조건부검증은검증정답을보고제외한모집단으로실제평가에이필터를적용불가. 필터된점수만으로실전성능개선주장금지. 원전체/2차전체손해도그대로보고. DIAG10만/노출된3seed/다른구성R3이며A/B·새holdout확증없어채택0/제출파일0.
- 계획/중간/최종독립비평 완료,critic소스수정요구의검증보강을실행: public와fold스냅샷NN거리·중앙값·삭제ID·X/Y원값 및24시간보존 검산,baseline raw혼합 및모든seedmean/prefix/clip·점수/fsum/bootstrap·모델영수증 재계산. CV개별candidate raw ET/LGB/MLP배열은미저장이라그구성원혼합 자체는소스검토한계; 최종9joblib에서는독립별도특징·모든구성원예측·혼합산술까지실행.
- root streaming CSV/fsum fresh재검산 PASS·공식train_y vspublic8640정답maxgap{r['official_vs_public_label_maxdiff']}. 독립최종9artifact비교maxgap{a['independent_vs_actual_prediction_maxdiff']} 및기존inmemoryprobe차{a['independent_vs_saved_probe_maxdiff']} PASS. root실제predict_bundle 경로public각farm×pass1/2 2일=8일192행/각층h5,h17 미래시간·다른농장 변조 및입력삭제불변,순서/평가weather불변 PASS.
- 훈련/검증오차격차·fold분산·농장×구간·seed별방향은최종독립결과/비평에저장. 최종400학습모델성능을새holdout에서측정한것은아님. 데이터프로파일전후20열/결측·dtype·키·시간·음수target·분포 및해시저장. 원본 수정/재번호부여0.

- 최종 비평의 범위 제한: 비평자는 크레딧 제한으로 마지막 산출물 파일을 재열람하지 못했다. 직접 검토한 앞선 코드/CV와 루트가 전달한 실제 최종 검사 결과로 메시지 비평을 완료했고 루트가 원문 기록했다. 상세는 critique_final_v1.md.
- ‘48시간 purge’ 표현은 같은 농장 ±1 상대 기록번호 제외로 정정한다. 인과성 v1 검사 실패는 두 구간 probe를 합친 입력의 이전 기록 수 assert 오류였으며 v2에서 정정 후 8검사 PASS. 모델/선정/점수 변경0. full_cleaning_recheck_v1.json에서 최종 400일 삭제 규칙·원자료 값·X/Y ID와 390일9360행도 독립 재검산 PASS. 상세 한계는 correction_evidence_v1.md.

### 다른 곳에 알릴 것
- 이번요청한새데이터/훈련모델완료. 실험후보이며제출구성확정0. 기존원66·타AI작업/프로세스건드리지않음.
- 조건부검증제외일과full최종삭제목록은reference가달라다를수있으며CVglobal목록누수금지.

### 다음/파일
- 새정제자료: 집/코덱스/local/ec_discord_clean_model_20261009_v1/dataset_full_train/train_X_clean_v1.csv 및 train_y_clean_v1.csv, removed_days_v1.csv/removed_row_ids_v1.csv/cleaning_manifest_v1.json.
- 저장모델: {modelpath}/9joblib·model_manifest_v1.json·fit_receipts_v1.json. 추론코드 {basepath}/predict_model_v1.py. 예측입력은각날h0부터대상시간까지prefix를제공하며공식test예측파일은생성하지않음.
- 재현순서 run_v2.py prepare→first→(midpoint검토)rest→final_fit,등록/이미생성파일보존하며재실행시새실험폴더버전을사용. 최종actualpath 추가검증 verify_model_causality_v2.py, 독립검산code/json·최종비평 모두같은analysis폴더.
- 같은날작업일지에세션2추가/HANDOFF갱신/카탈로그새6.{n}. gitcommit/push0. 모델/CSV큰산출물은local Drive동기화,두AI완료뒤sync_end1회.
'''
with (ROOT/'집/코덱스/작업일지/2026-10-09.md').open('a',encoding='utf8') as out:out.write(text)
row=f'''\n| 6.{n} | **고EC 불일치일 전체 삭제 정제자료 + 전구성원 재학습R3 모델 제작** (2026-10-09 집 코덱스, 사용자지시/3단계독립비평) | CV공개360/DIAG10×3seed,ET/LGB/MLP각30=90fit·baseline replay3;5NNproxy 학습일mean>=1·이웃중앙보다>.5·자기/±1제외,fold별선정/season·imputer·scaler·bounds모두cleanfit/global목록유입0. 동일정제q {G[('filtered','BASE','ensemble')]['days']}일 R3 .6/.3/.1 기준{f('filtered','BASE'):.6f}→clean{f('filtered','CLEAN'):.6f}({pct('filtered'):+.3f}%,p={probs['filtered']});원q전체360일{f('all','BASE'):.6f}→{f('all','CLEAN'):.6f}({pct('all'):+.3f}%,p={probs['all']})/공동alpha.0125/条件부점수실전주장금지. 최종CV후에만400공식일정답포함, {d['removed_days']}일{d['removed_rows']}행삭제→{d['after']['days']}일{d['after']['rows']}행 새XY 및전체9joblib학습;40재검증0. dataset/NN·산술·bootstrap·실제192행각farm×pass/h5,h17인과성·독립최종9구성원pred PASS. 모델은별도R3(PFN/SG2없음)/채택·제출0/A·B확증없음 | {basepath}/PLAN_v1.md;registration_v1.json;final_score_v1.json;root_final_cv_recheck_v1.json;final_artifact_independent_v1.json;final_model_causality_v2.json;critique_final_v1.md;{modelpath}/model_manifest_v1.json |\n'''.replace('条件부','조건부')
assert cat.read_text(encoding='utf-8-sig')==old
with cat.open('a',encoding='utf8') as out:out.write(row)
handoff=ROOT/'공용/HANDOFF.md';previous=handoff.read_text(encoding='utf-8-sig')
head=f'> **2026-10-09 · 집 · 코덱스 고EC 통삭제 데이터/새모델 제작 완료:** 사용자지시 새R3(.6ET+.3LGB+.1MLP,3seed) 전구성원/전처리clean재fit. CV90모델·baseline3재현·최종9저장,공개360global12일제거348일8352행/최종CV후공식400일{d["removed_days"]}일제거→{d["after"]["days"]}일{d["after"]["rows"]}행 새XY. 동일조건부검증{G[("filtered","BASE","ensemble")]["days"]}일 {f("filtered","BASE"):.6f}→{f("filtered","CLEAN"):.6f}({pct("filtered"):+.3f}%,p {probs["filtered"]})·원全360日 {f("all","BASE"):.6f}→{f("all","CLEAN"):.6f}({pct("all"):+.3f}%,p {probs["all"]}). 조건부q는정답알고제외한모집단으로실전개선주장불가/DIAG만·A/B없음/모델생성만·채택/제출0. foldtrain내선정/global리스트CV0/40일최종fit만·재검증0. 데이터/NN/산술/bootstrap/rootfsum/독립9artifact 및192행각farm×pass/h5,h17 미래·타farm변조/삭제불변PASS. 카탈로그6.{n},집/코덱스/작업일지/2026-10-09.md세션2, own {basepath}/ 및 local/ec_discord_clean_model_20261009_v1. 모델/데이터local Drive대상,원66미완료유지/두AI완료뒤sync_end1회.\n'
head=head.replace('全360日','전체360일')
handoff.write_text(head+previous,encoding='utf8')
print(json.dumps(dict(catalogue=f'6.{n}',removed_days=d['removed_days'],clean_days=d['after']['days'],clean_rows=d['after']['rows'],filtered_pct=pct('filtered'),all_pct=pct('all'),model=str(L/'model_full_clean_v1/model_manifest_v1.json')),ensure_ascii=False))
