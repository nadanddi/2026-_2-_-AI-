import json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
assert (H/'critique_final_v1.md').exists() and (H/'correction_evidence_v1.md').exists()
s=json.loads((H/'final_score_v1.json').read_text(encoding='utf8'))
a=json.loads((H/'final_independent_v5.json').read_text(encoding='utf8'))
r=json.loads((H/'root_final_recheck_v1.json').read_text(encoding='utf8'))
assert a['status']=='INDEPENDENT_CHECK_PASS' and r['status']=='PASS' and r['candidate_fits']==90
G={(g['scope'],g['arm'],g['seed']):g for g in s['groups']}
get=lambda scope,arm,seed='ensemble':G[(scope,arm,seed)]['rmse']
pct=lambda scope,arm:100*(get(scope,arm)/get(scope,'BASE')-1)
alpha=.025/3
bs={(b['arm'],b['reference']):b['p_worse'] for b in s['bootstrap']}
all_same=all(get('normal','ALL_HIGH',str(seed))<get('normal','BASE',str(seed)) for seed in (7,101,2024))
dis_same=all(get('normal','DISCORD_HIGH',str(seed))<get('normal',ref,str(seed)) for seed in (7,101,2024) for ref in ('BASE','CONTROL'))
assert all_same and bs[('ALL_HIGH','BASE')]<alpha
assert dis_same and bs[('DISCORD_HIGH','BASE')]>=alpha
assert pct('all','ALL_HIGH')>=2 and pct('all','DISCORD_HIGH')>=2
cat=ROOT/'공용/데이터_단서_카탈로그.md';catalog=cat.read_text(encoding='utf-8-sig')
last=max(int(n) for n in re.findall(r'^\| 6\.(\d+) \|',catalog,re.M));c1=last+1;c2=last+2
path='집/코덱스/analysis/ec_high_training_hypothesis_20261009_v1'
table='| 학습 제거 | 일반 329일 RMSE | 고EC 31일 RMSE | 전체 360일 RMSE | 2차 일반 41일 RMSE |\n|---|---:|---:|---:|---:|\n'
for arm,label in [('BASE','기준 EC14'),('ALL_HIGH','고EC 전체'),('DISCORD_HIGH','입력 이웃 대비 고EC'),('CONTROL','같은 수 일반일')]:
    table+=f"| {label} | {get('normal',arm):.9f} | {get('high',arm):.9f} | {get('all',arm):.9f} | {get('pass2_normal',arm):.9f} |\n"
text=f'''# 2026-10-09 — 집 / 코덱스

## 읽은 것
- 사용자 고EC 가설 및 ‘실험해봐’, 세션 시작 문서·HANDOFF·공용 메모리와 다른 세 폴더 일지 검토(전날 일지 참고).
- skill-not-luck / analysis-before-answers / harsh-critic-every-result / rule5-interpretation, predictive-modeling / analysis-verification 스킬.
- EC14 등록 DIAG10·현재47열·기존 기준30개 ET/LGB/MLP와 PFN40개 캐시/입력 서명. 평가 전체행 RMSE 공식근거는 비평가가 PDF 읽기전용 추출·페이지 시각 확인.

## 한 것
- 후보fit 전 계획v1→v2/v3 보완 및 independent critic 계획/중간/최종 검토. 공개360일 DIAG10 10fold×3seed(7/101/2024), 후보ET90fit+기준ET1fit 재현. 기존 검증 날짜·정답은 그대로.
- 고EC=학습일 EC 일평균>=1. ALL 전체 제거, DIS 같은 온실/±1기록 제외 입력5NN EC중앙보다 .5 초과인 고EC 제거, CONTROL 농장×구간별 DIS와 같은 수 일반일 무작위 제거(고정 seed20261009). fold별 ALL20~26일/DIS8~13일.
- ET만 개입. LGB/MLP/PFN·계절변수·SG2 참조정답·clip범위는 고정. 따라서 전체 멤버 학습자료 제거 실험이 아님.
- 실행 코드 run_v4.py, 등록 registration_v4.json. early/late를 순차 실행, 90fit 종료/10fold CSV+영수증 완료. 새 제출 파일·채택·A/B·test 예측 생성0.
- 별도 사전 계획 PLAN_lb_v1.md: 기존9회차 제출CSV와 고정 EC1 기준만으로 평가 고EC 부재 가설의 집계 Jensen 하한 계산. 공개점수로 모델/임계값/가중치 선택0, 개별 숨은 정답 역산0.

## 결과
{table}
- 카탈로그 **6.{c1}**: ALL 일반 RMSE {pct('normal','ALL_HIGH'):.3f}%·모든seed 개선, 농장층화 day//5 기록번호블록 bootstrap20000 p_worse=.0053<.025/3=.008333으로 일반일 진단 screen 통과. 그러나 전체 {pct('all','ALL_HIGH'):+.3f}%·고EC {pct('high','ALL_HIGH'):+.3f}% 악화, 사전 전체/2차일반 +2% 효용guard 실패. DIS 일반 {pct('normal','DISCORD_HIGH'):.3f}%·모든seed 개선, 기준 대비 p=.0143으로 screen 실패(대조 대비 .00285는 대체불가), 전체 {pct('all','DISCORD_HIGH'):+.3f}%·고EC {pct('high','DISCORD_HIGH'):+.3f}% 악화. 두 제거안 채택0.
- 카탈로그 **6.{c2}**: 실제9회차CSV SHA f2fe5cf7e5299c2c0f2ac39f0e071066744b0c75ea19e1d7dff1e3a6282ba18a, 60일×24행. 모든 실제 평가일 일평균EC<=1이면 전체균등RMSE>=.21869200132495553이어야 함. 실제EC점수.1384의 보수상한.1385보다 큼. 따라서 보관CSV↔점수 연결을 전제하면 일평균EC>1 평가일 존재. ‘설명되지 않는 고EC’의 존재/부재나 라벨오류는 이 하한으로 구분 불가.
- 루트 streaming CSV/math.fsum 재계산(root_final_recheck_v1.py)과 별도 비평가 독립검산(verify_independent_v5.py)으로 핵심 수치/144그룹4지표/3bootstrap 및 핀·제거·fit영수증 확인. 기준ET raw 재현차0, 기준최종 최대차4.440892098500626e-16. LB는 float/fsum·일합계식·Decimal60/65자리 일치.
- 유효성 한계: 이미 노출된 DIAG10만/ET경로만/단회 일반일 대조/실제 달력과 다른 기록번호블록. 학습 정답분포 이동과 ‘설명 불가’ 특이성을 분리 못함. 정상일 이득도 소수 날에 집중(최종 독립 보충표 참고), 모든 날/모든 검증기 효과를 주장하지 않음. 95% CI는 서술용, 본페로니 조정 CI가 아님.
- 메타데이터 예외: 기존 DIAG10_9_pfn_3.json 3422바이트가 모두 NUL. 원파일 보존. 구 동결 preparation의 NPZ SHA·실제 문맥index/row_id·현재 입력/train/query 서명으로 대체 검증, 그 손상 JSON 자체의 직접 provenance 재검증은 불가(사전 비평 승인). run_v1/v2/v3와 실패 로그도 보존.
- 독립결과v5의 ‘calendar blocks’ 설명은 잘못된 문구이며 correction_evidence_v1.md로 명시 정정. 실제 계산은 사전 고정된 day//5 기록번호 구간 그대로, 수치 변경0.

## 다른 곳에 알릴 것
- 일반일 개선 신호와 전체 성능 손해를 구분할 것. 고EC 제거를 평가에 고EC가 없다는 전제로 채택하면 안 됨. EC14 유지.
- LB는 고정1 기준 집계 가설 진단이며 향후 모델 선택/계수 조정 근거로 쓰지 않음.
- 공유 저장소에 다른 AI의 research/logs 변경 및 다른 Python 작업도 있음. 건드리거나 종료하지 않음. git commit/push0.

## 다음
- 이번 가설 진단은 완료. ‘설명되지 않는 고EC’라는 물리적 구분은 미입증, 현 삭제안은 효용 실패. 추가 연구 시 새 사전설계/독립 검증부터, 같은 결과에 맞춰 임계값 조정 금지.
- 기존 원66 작업 미완료 상태는 변경하지 않음. 두 AI 완료 뒤 사용자가 sync_end 1회 실행.

## 파일
- 코드·계획·등록·비평·최종 결과: {path}/ (final_score_v1.json, final_groups_v1.csv, final_independent_v5.json, root_final_recheck_v1.json, critique_final_v1.md, correction_evidence_v1.md, evaluation_absence_bound_v1.json, critique_lb_result_v1.md 등).
- 무거운10fold 예측/영수증: 집/코덱스/local/ec_high_training_hypothesis_20261009_v1/folds/ (Drive 동기화 대상).
- 공용/HANDOFF.md 갱신, 공용/데이터_단서_카탈로그.md 새6.{c1}·6.{c2} 추가.
'''
log=ROOT/'집/코덱스/작업일지/2026-10-09.md';assert not log.exists()
with log.open('x',encoding='utf8') as f:f.write(text)
rows=f'''\n| 6.{c1} | **고EC 학습일 ET 제거 진단: 일반일 개선은 있으나 전체효용 실패, 삭제안 채택0** (2026-10-09 집 코덱스, 3단계 독립 비평) | 현재EC14 DIAG10 10fold×3seed/공개360일, ET후보90fit. ALL 일반329일 .110631→.092924({pct('normal','ALL_HIGH'):.3f}%, 모든seed/p_worse .0053<.008333 screen통과), 고EC31일 .491951→.667284·전체 .178957→.215020({pct('all','ALL_HIGH'):+.3f}%)로+2%효용guard실패. DIS 5NN대비.5초과고EC만제거: 일반.102701/p .0143로screen실패(일반일삭제대조.111280 대비p .00285가대체불가), 전체.194196({pct('all','DISCORD_HIGH'):+.3f}%)로guard실패. ET만개입/나머지멤버·SG2ref/bounds고정; 일반일target분포 이동과discord특이성 분리불가/DIAG만/no adoption/no submission. baseline raw0·final≤4.44e-16, 독립144그룹/3bootstrap PASS. day//5는5기록번호구간(달력아님), 손상PFN메타1개는 동결SHA/문맥/입력서명대체검증 | {path}/run_v4.py; final_score_v1.json; final_independent_v5.json; root_final_recheck_v1.json; critique_final_v1.md; correction_evidence_v1.md |\n| 6.{c2} | **평가에 고EC일이 전혀 없다는 가정(일평균EC<=1)은9회차 예측·공식점수와 모순** (2026-10-09 집 코덱스, 사전계획·독립 검산) | 실제9회CSV f2fe5cf7e529…/1440행60일×24, 모두 실제일평균<=1이면 Jensen RMSE하한 sqrt(mean(max(제출일평균−1,0)^2))=.21869200132495553. 점수.1384의보수상한.1385 초과. 공식문제설명서4쪽7.1 n1440 전체균등RMSE·7.3 별도비공개없음/현행중간안내변경0. 보관CSV↔공식score연결 전제하에 최소1일의실제일평균>1 존재; 어느날/몇날/설명불가고EC인지/라벨오류는추론불가. 임계1 사전고정/스캔·개별정답역산·모델조정0, 공개점수 집계진단만 | {path}/PLAN_lb_v1.md; evaluation_absence_bound_v1.py; evaluation_absence_bound_v1.json; evaluation_absence_bound_independent_v1.json; critique_lb_result_v1.md |\n'''
assert cat.read_text(encoding='utf-8-sig')==catalog
with cat.open('a',encoding='utf8') as f:f.write(rows)
handoff=ROOT/'공용/HANDOFF.md';old=handoff.read_text(encoding='utf-8-sig')
update=f'> **2026-10-09 · 집 · 코덱스 고EC 가설 진단 완료:** DIAG10 10fold×3seed·候補ET90fit+기준再現1fit/8640행360일. ALL 일반329일 .110631→.092924({pct("normal","ALL_HIGH"):.3f}%, 전seed/p .0053로일반screen통과)이나 전체 .178957→.215020({pct("all","ALL_HIGH"):+.3f}%)·고EC .491951→.667284로효용guard실패. DIS 이웃대비고EC선별 제거 일반.102701/p .0143 screen실패·전체.194196({pct("all","DISCORD_HIGH"):+.3f}%). 두안 채택·제출0/EC14유지. ET경로만/나머지멤버·SG2ref/bounds고정/이미본DIAG만; 원인·라벨오류·설명불가특이성 미입증. 사전/중간/최종혹독비평+루트fsum·144그룹/3bootstrap 독립PASS, 기준 raw0/final≤4.44e-16. 별도집계하한: 모든평가일 EC일평균<=1이면RMSE≥.218692인데9회점수.1384(보수상한.1385), 공식1440행전체RMSE 지원→보관CSV/score연결전제하에高EC평가일존재(설명불가날인지는미상). 카탈로그6.{c1}/6.{c2}, 집/코덱스/작업일지/2026-10-09.md 및 {path}. 이번진단worker종료/원66미완료유지/다른Python작업건드리지않음. 다음:삭제안비추천·추가연구새사전설계부터. 두AI끝난뒤sync_end1회.\n'
update=update.replace('候補','후보 ').replace('再現','재현 ').replace('高EC','고EC')
assert handoff.read_text(encoding='utf-8-sig')==old
handoff.write_text(update+old,encoding='utf8')
print(json.dumps(dict(worklog=str(log),catalogue=[f'6.{c1}',f'6.{c2}'],all_normal_pct=pct('normal','ALL_HIGH'),all_total_pct=pct('all','ALL_HIGH'),dis_normal_pct=pct('normal','DISCORD_HIGH'),dis_total_pct=pct('all','DISCORD_HIGH'),adoption=False),ensure_ascii=False))
