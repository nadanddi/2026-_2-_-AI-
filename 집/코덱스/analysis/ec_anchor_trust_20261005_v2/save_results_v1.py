from pathlib import Path
import sys,json,csv,re,hashlib,zipfile
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
results={m:load(H/f'verification_{m}_v1.json') for m in ['GATE','RIDGE']}
assert all(v['status']=='PASS_ARITHMETIC_REPLAY' and v['decision']=='REJECT' for v in results.values())
assert all(load(H/f'crosscheck_{m}_v1.json')['status']=='PASS_DECIMAL30_MANUAL_BOOT3' for m in results)
assert load(H/'diagnosis_crosscheck_v1.json')['status']=='PASS_NUMPY_PANDAS_VS_SCALAR'
diag=load(H/'case_diagnosis_v2.json');rows=[];summary={}
for mode,v in results.items():
 for r in v['scores']:rows.append(dict(mode=mode,**r))
 fits=load(H/f'fit_{mode}_v1.json');meta=[load(ROOT/'집/코덱스/local'/H.name/mode/f"{r['validator']}_{r['fold']}_{r['seed']}_fit.json") for r in fits['manifest']]
 summary[mode]=dict(diag=[s['change_pct'] for s in v['scores'] if s['validator']=='DIAG10'],ordinary=[s['change_pct'] for s in v['segments'] if s['segment']=='ordinary'],high=[s['change_pct'] for s in v['segments'] if s['segment']=='high'],pass2=[s['change_pct'] for s in v['segments'] if s['segment']=='pass2'],empty_cells=sum(m['n']==0 for m in meta),native_models=sum(not m['model_none'] for m in meta),improved=sum(r['candidate']<r['baseline'] for r in v['scores']),p_worse={s:b['p_worse'] for s,b in v['bootstrap'].items()})
with (H/'comparison_scores_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fmt(a):return ', '.join(f'{v:+.4f}%' for v in a)
text=f'''# EC 신뢰도 순차 실험 결과 — 2026-10-05 04:55 집 코덱스

사용자 요청에 따라 중요도1의 이웃보정 신뢰도와 제한된 보정량 학습을 순차 실행했다. 학습 피처의 자기참조/시점 문제와 실제v2 기준선 연결을 두 실험의 전제 조건으로 처리했고, 완료 후 중요도2의 세 실패 날짜를 재진단했다. 현재 EC 계절v2를 유지한다. 새 채택·제출0이다.

| 실험 | DIAG10 RMSE 변화(7/101/2024) | 일반329일 변화 | 고EC31일 변화 | 판정 |
|---|---|---|---|---|
| 가족25 비용가중 신뢰도 GATE | {fmt(summary['GATE']['diag'])} | {fmt(summary['GATE']['ordinary'])} | {fmt(summary['GATE']['high'])} | 기각 |
| 가족26 제한적 잔차 RIDGE | {fmt(summary['RIDGE']['diag'])} | {fmt(summary['RIDGE']['ordinary'])} | {fmt(summary['RIDGE']['high'])} | 기각 |

각각22fold×3seed=66셀/83160행,5검증기×3seed=15점수 전수 검산을 마쳤다. 엄격 개선은 둘 다0/15다. A/B/EXT12에서는 baseline과 동일하며 동일은 개선PASS가 아니다. GATE DIAG p_worse={summary['GATE']['p_worse']}, RIDGE={summary['RIDGE']['p_worse']}. 고정 alpha=.025/26 및 조정CI 상한<0을 통과하지 못했다. 반복 공개 검증이고 독립 미사용holdout이 아니므로 통과하더라도 즉시확정 구성으로 만들지 않기로 사전등록했다.

## 중요한 새 관찰

F13 231·233, F47 216의 세 날짜는 두 실험 모두3시드에서 제곱오차가 감소했다(총18개 case×mode×seed 비교). 사례 손해를 줄이는 보정은 만들 수 있었다. 다만 다른 날 손해가 더 컸으므로 성공 사례만 근거로 채택하지 않는다.

GATE가 실제 고EC 날에서 수정한 시간행은 세시드 합산153개이고 **전부 EC를 낮췄다**(독립 NumPy/pandas↔scalar84검사PASS). 고EC RMSE 전체는 위 표대로 악화했다. 이는 기존 높은EC 과소예측을 더 키우는 방향의 보정이 생긴 관찰이며, 모든수정행이 개별적으로 악화했다는뜻은 아니다. 신뢰도높음: 관측산술. 왜그런지를 입력정보부재/다른동 원인으로 확정하는신뢰도는 낮음.

RIDGE는 하루별 이분법 대신 현재시각의 조건부 잔차를 추정하고 보정폭을±.06으로 제한했다. GATE보다 손해는 작지만 개선은 없었다. GATE .5clip(anchor−prefix,±.6), RIDGE .2clip(residual,±.3)은 실행 전에 둘 다등록했고 GATE결과로RIDGE를튜닝하지않았다.

## 입력·학습·검증을 어떻게 바꿨나

- 실제현재v2의 원형 .8R3+.2PFN4bag→shrink.5→train범위clip 출력을 동일기준선으로 유지했다. 새 보정은 **그 최종출력 뒤** 더하고 train범위clip한다. SG2재현이나rawmix단계교체 실험이 아니다.
- 이웃은 같은농장·같은pass의앞선공개학습날짜만. 서명은 현재시각까지의 실내3+제어7+외기4 평균이다. 결측대체/거리표준화는reference-hour 행만 fit하며 query는상태를갱신하지않는다. 원EC정답/test값/잠금/EL1재채점0.
- 기존 matched inner에서 a로학습한 baseline→b 예측과a의이웃으로 메타학습. 외부q는tr의이웃을쓰며 둘다 validation±1일purge/orderedID/hash를검사했다. b 자체정답은학습목적으로만, 외부정답은score에서만사용했다. 완전한outer-tr OOF가아닌단일heldout b하위집합이라는한계가있다.
- 첫v1 실행은eligible0인 DIAG fold7에서21셀후중단했다. 부분파일·실패로그보존,외부score0에서 빈집합이면 baseline유지로새v2에규칙을추가하고 c9337f0에재등록했다. 최초규칙과완전히동일한실행이라고표현하지않는다. 참조범위/미래제한/검증기/문턱은완화하지않았다.
- 빈학습셀: GATE {summary['GATE']['empty_cells']}/66, RIDGE {summary['RIDGE']['empty_cells']}/66. 실제추정기 학습셀: GATE {summary['GATE']['native_models']}, RIDGE {summary['RIDGE']['native_models']}. 후보정보/학습자료가없는폴드를baseline유지했으므로 '모든검증기에서새모델을동일하게학습했다'고주장하지않는다.

## 독립 검증과 반론

원본캐시/provenance는 이전불변preparation SHA와모든파일SHA/순서/target/범위의감사증거를재사용했고 R3/PFN 원모델을 재학습하지않았다. 처음각mode 모델은freshfit재현, 계수재생,단독행/순서/다른query/scalar clipping, rawprefix절단/미래변조6건감사PASS. 별도learning검산은scaler순수fsum/표본ID/피처·costSHA/목적함수gradient를검사하고 whole앞에결속했다. Whole는exact66키/CSVschema/모든key·bounds·anchor·baseline·labels·계수재생·aggregateSHA 검사후점수계산했다. 독립Decimal30RMSE/mannual3bootstrap도일치했다.

반론: 새gate의실패가신뢰도모델자체의무효를증명하는가? 아니다. baseline위에 새earlier-day이웃생성+gate를함께연결한pipeline 실험이므로 SG2의신뢰도학습만을분리한효과라고볼수없다. 이전A/B가동일한것도현재pass2적용/빈자료설계제약에따른것이며 입력정보부재의증거가아니다.

다음실험의선행과제는 현재시점에 실제고EC를유지할신호와 과대예측을내릴신호를분리해 검사하고, 메타학습이 가능한pass2 내부OOF표본을확보하는것이다. 세날ID별예외/외부정답으로문턱선택은하지않는다. Claude HG2가보호게이트를별도로수행중이므로중복하지않는다.

## 클로드 완료 시점 갱신

GATE완료뒤와RIDGE완료뒤 Claude 1fd45d6→04394fc/c013438/d1e6691/4303eb3를읽었다. FZ0/FZ1 코드·완료로그·카탈로그6.320/321 및HG2 prereg/진행로그를조회했다. FZ0/FZ1 수치는작성자로그출처이고 이번에독립재채점하지않았다. 특정L2 logistic의full-day AUC를최적구분성능의상한이나정보부재증명으로보는6.320해석은보류한다. FZ1도작성자사전AUC단서실패이며HG2는새시드/새배치/일반보호조건으로확인중이다. HG2는미완료·중복학습0.

TabDPT는동일PID9496/10-04 15:25:38/run_v4.py로4/66로그완료를확인했다. 최초감사전부분점수0·재시작0·monitor유지·전체EC목표미완료.

재현파일: run_v6.py, preparation_v6.json, registration_v1.json, verify_learning_v4.py, verify_v5.py, crosscheck_v1.py, diagnosis와独立비평. 공개예측CSV는 집/코덱스/local/ec_anchor_trust_20261005_v2/GATE·RIDGE/oof.csv, 비교점수 comparison_scores_v1.csv다. 이는실험출력이고test/제출물은없다.
'''
with (H/'실험결과_v1.md').open('x',encoding='utf-8') as f:f.write(text)
sources=['fz0_false_gate_forensics_v1.py','fz0_false_gate_forensics_v1.log','fz1_domain_score_and_dip_timing_v1.py','fz1_domain_score_and_dip_timing_v1.log','ec3_HG2_domain_guard_gate_v1.py','ec3_HG2_domain_guard_gate_v1.log']
with (H/'claude_refresh_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(last_claude_commit='4303eb3',sources={n:sha(ROOT/'집/클로드/research'/n) for n in sources},HG2='IN_PROGRESS_UNSCORED',numbers='author log attribution only; no new rescore',FZ0_inference='information absence not established by selected classifier'),f,indent=2)
catalog=ROOT/'공용/데이터_단서_카탈로그.md';t=catalog.read_text(encoding='utf-8');n=max(int(v) for v in re.findall(r'\| 6\.(\d+) \|',t))+1
entry=f'| 6.{n} | **EC 가족25 이웃 신뢰도·26 제한잔차 순차실험 완료/기각** (집 코덱스10-05) | actualv2기준 각66셀83160행/strict15 개선0. GATE DIAG +3.079~3.517%, 일반−1.420~−1.946%/고EC+5.664~6.140%;RIDGE DIAG+.368~.435%. 두mode모두F13 231/233·F47 216을3seed에서개선(18case),그러나다른고EC손실로전체악화;GATE고EC수정153행(seed합산)전부하향. alpha.025/26·원규칙유지,learning/scaler-gradient·whole/Decimal30/manualboot3·case84独立PASS. 初실행eligible0/partial21셀score0보존후新v2 baseline fallback 사전c933등록;현재v2유지/채택0/제출0. 全OOF아닌heldout-b 한계/SG2순수gate효과아님 | 집/코덱스/analysis/ec_anchor_trust_20261005_v2/실험결과_v1.md |\n'
entry2=f'| 6.{n+1} | **Claude FZ0/FZ1/HG2 갱신: classifier 실패는 정보부재 증명이 아님** (집 코덱스10-05) | 1fd45d6이후4303eb3까지 code/log readonly. 6.320의full-day L2 logistic AUC는모델특정성과소표본 한계가있어최적성능상한/정보부재확정이라는해석을지지하지않음. FZ1作者단서실패,H G2 새seed37/1212/4242·DIAG10q·일반보호조건확인진행중;중복0/공개재채점0/EL1재채점0. 수치출처作者로그이고독립산술확인이라고표현하지않음 | 집/코덱스/analysis/ec_anchor_trust_20261005_v2/claude_refresh_v1.json, 실험결과_v1.md |\n'
catalog.write_text(t+'\n'+entry+entry2,encoding='utf-8')
notice=f'2026-10-05 04:55 집 코덱스: 가족25GATE/26RIDGE 순차 실제완료·각66셀83160행/strict15 0개개선·기각. DIAG GATE+3.079~3.517%,RIDGE+.368~.435%;두mode모두오탐3날×3seed개선이나고EC손실우세. source前登録869e00a/빈inner score0보완c9337f0/partial21보존;learning/whole/Decimalmanual/diagnostic독립PASS. 새보고서 ec_anchor_trust_20261005_v2/실험결과_v1.md·6.{n}/{n+1},actualv2유지/제출0. Claude4303eb3까지FZ0/1/HG2갱신(作者수치로그출처/재채점0·정보부재확정반론/HG2진행중중복0). TabDPT同PID9496開始10-04 15:25:38·4/66·부분점수/재시작0/monitor유지/전체EC목표미완료.'
p=ROOT/'공용/HANDOFF.md';t=p.read_text(encoding='utf-8');head,tail=t.split('\n',1);p.write_text(head+'\n\n> **'+notice+'**\n'+tail,encoding='utf-8')
p=ROOT/'공용/확인기록.md';ls=p.read_text(encoding='utf-8').splitlines();ls=['| 집 · 코덱스 | 2026-10-05 | '+notice+' 이전검토유지·타AI채팅메시지0. |' if x.startswith('| 집 · 코덱스 |') else x for x in ls];p.write_text('\n'.join(ls)+'\n',encoding='utf-8')
p=ROOT/'집/코덱스/작업일지/2026-10-05.md';t=p.read_text(encoding='utf-8');p.write_text(t+'\n\n## 04:55 중요순서 실험 실제완료\n\n- '+notice+'\n- actual입력/피처시점/참조purge/inner b자기참조 제거를먼저검사한뒤신뢰도gate→제한ridge 순차실행. 3실패날이둘다개선돼도전체고EC악화가우세해서채택하지않음.\n- v1~6 준비보완/초기NaN·경로치환/empty actual실패로그를보존. 최종새v2의각66cell·원baseline·model계수재생·scaler·eligible/cost fingerprints·objectivegradient·bootstrap검산완료.\n- 파일: 새v2 report/scoreCSV/reproZIP·case/검증JSON·원local GATE/RIDGE OOF와fit JSON. 다른AI폴더수정/원EC/test/lock/EL1재채점0.\n- 다음: HG2 완료readonly갱신, pass2 meta용matched crossfit확보/실제고EC 보호신호 검증. family20 完全66 최초감사PASS 후원whole_v2·재시작0.\n',encoding='utf-8')
with (H/'saved_summary_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(models=summary,catalog=[f'6.{n}',f'6.{n+1}'],report_sha=sha(H/'실험결과_v1.md')),f,ensure_ascii=False,indent=2)
readme='실험 재현 묶음. main c9337f0 및 같은 Python 환경과 Drive 기존 캐시가 필요합니다. 저장된 GATE/RIDGE 출력은 재학습하지 않습니다. 각 mode별 verify_learning_v4.py -> verify_v5.py -> crosscheck_v1.py로 검산합니다. 검사 결과는 이미 존재하므로 기존 파일을 덮어쓰지 말고 새작업폴더/새결과버전을 준비하세요. 새로운 모델 fit 재현은새이름으로 --prepare 후 등록과 --mode 순서. 원입력/의존코드/기존큰캐시는repo/Drive에서 공급하며 이ZIP은독립standalone환경이아닙니다. test 예측/제출물0.'
with (H/'재현_README_v1.txt').open('x',encoding='utf-8') as f:f.write(readme)
with zipfile.ZipFile(H/'실험재현_v1.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in H.iterdir():
  if p.is_file() and p.suffix in ['.py','.json','.md','.csv','.txt']:z.write(p,str(p.relative_to(ROOT)))
print(json.dumps(summary,ensure_ascii=False));print('saved catalog',n,n+1)
