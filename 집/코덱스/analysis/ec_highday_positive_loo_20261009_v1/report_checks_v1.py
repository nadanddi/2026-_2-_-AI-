"""Independent scalar arithmetic and a Korean positive-LOO report."""
from pathlib import Path
import sys,json,csv,math
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(ROOT/'집/코덱스/analysis/ec_highday_classifier_20261009_v1'))
import runtime_v1,run_v1 as P
L=ROOT/'집/코덱스/local'/H.name
s=json.loads((H/'final_score_v1.json').read_text(encoding='utf8'));assert s['status']=='FULL_LOO'
rows=[]
for p in sorted((L/'predictions').glob('*_v1.csv')):
 with p.open(encoding='utf8') as f:rows+=list(csv.DictReader(f))
assert len(rows)==26*24 and len(set(r['row_id'] for r in rows))==len(rows)
fresh=[]
for hour in [0,8,15,23]:
 rr=[r for r in rows if int(r['hour'])==hour];scores=[float(r['ensemble']) for r in rr]
 caught=sum(p>=.5 for p in scores);missed=sum(p<.5 for p in scores);caught02=sum(p>=.2 for p in scores);b=math.fsum((p-1)**2 for p in scores)/len(scores);mean=math.fsum(scores)/len(scores)
 t=next(t for t in s['groups'] if (t['hour'],t['arm'])==(hour,'ensemble'))
 assert len(rr)==26 and [caught,missed,caught02]==[t['caught05'],t['missed05'],t['caught02']] and abs(b-t['conditional_positive_brier'])<1e-12 and abs(mean-t['mean_score'])<1e-12
 fresh.append(dict(hour=hour,n=len(scores),caught05=caught,missed05=missed,caught02=caught02,positive_brier=b,mean_score=mean))
with (L/'final_paired_h15_v1.csv').open(encoding='utf8') as f:paired=list(csv.DictReader(f))
gained=sum(float(r['ensemble'])>=.5 and float(r['diag_ensemble'])<.5 for r in paired);lost=sum(float(r['ensemble'])<.5 and float(r['diag_ensemble'])>=.5 for r in paired);delta=math.fsum((float(r['ensemble'])-1)**2-(float(r['diag_ensemble'])-1)**2 for r in paired)/len(paired)
assert gained==s['paired_h15']['gained'] and lost==s['paired_h15']['lost'] and abs(delta-s['paired_h15']['delta_positive_brier'])<1e-12
P.save(H/'fresh_scalar_checks_v1.json',dict(status='PASS',hour_metrics=fresh,gained=gained,lost=lost,positive_brier_delta=delta,extra40_rescored=False))
lines=['# 고EC일 한 날씩 제외하는 분류 진단', '', '사용자제안에따라공개고EC26일을일단위LOOCV로검증했다. 고EC는일평균sub_ec≥1.2이며학습에는일반일과고EC를모두포함했다. 제외일24행을모두빼고같은농장±1/다른농장±3/소비40일주변도제외했다. 기존73prefix특징과ET3seed평균·logit/prior기준은그대로유지했다. 일반일은검증하지않았으므로오탐률·precision·AUC/AP·전체구분정확도를이결과에서계산하지않는다.', '', '## 탐지 결과', '', '|당일 입력 시각|고EC일|탐지(.5)|놓침|재현율|탐지(.2 보조)|', '|---|---:|---:|---:|---:|---:|']
for t in fresh:lines.append(f"|{t['hour']}시까지|26|{t['caught05']}|{t['missed05']}|{t['caught05']/26:.1%}|{t['caught02']}|")
x=s['paired_h15'];lines+=['', '## 기존 묶음 검증과 같은26일 비교', '', f"15시: 기존DIAG {x['diag']['caught05']}/26 탐지 → 새LOO {x['loo']['caught05']}/26. 새로잡은날{x['gained']}·기존에잡다가놓친날{x['lost']}. 고EC조건부Brier {x['diag']['conditional_positive_brier']:.6f}→{x['loo']['conditional_positive_brier']:.6f};이는양성날에만계산한오차로전체분류Brier와다르다.", '', f"각LOO학습에는 {s['train_days']['min']}~{s['train_days']['max']}일(평균{s['train_days']['mean']:.2f}),고EC{s['train_high']['min']}~{s['train_high']['max']}일(평균{s['train_high']['mean']:.2f})이남았다. purge때문에항상나머지고EC25일을다학습한것은아니다. 기존DIAG의해당일학습고EC는13~20일이었다. 모든LOO학습집합은해당기존DIAG학습집합의superset임을training_exposure_v1.json으로확인했다. 일반일학습량도동시에늘어고EC표본수효과만분리할수없다.", '', f"탐색농장층화5기록번호블록 {x['blocks']}개·20000회bootstrap: 양성Brier차이 CI95={x['delta_brier_ci95']},p_worse={x['p_worse']:.5f}. 새고EC자료·미사용홀드아웃이없고training겹침·사건내날의존성한계가있다. 이값은자료내paired진단이며채택판정이나독립확증에쓰지않았다.", '', '## 시드·농장·구간', '']
for t in s['groups']:
 if t['hour']==15:lines.append(f"- {t['arm']}: {t['caught05']}/26 탐지, .2탐지{t['caught02']},고EC조건부Brier{t['conditional_positive_brier']:.6f}.")
for t in s['segments']:lines.append(f"- {t['farm']} 뒤구간={t['pass2']}: {t['days']}고EC일,기존{t['diag']['caught05']}→LOO{t['caught05']}탐지,놓침{t['missed05']}.")
lines+=['', '## 해석과 검증', '', '탐지수·학습일수의산술신뢰도는높음: rawCSV별도loop합계로각시각counts/mean/positiveBrier와pairedgain/loss를재계산했고, 독립비평가가원metadata/26purge분할/라벨·시드/실제예측파일/집계·bootstrap을검산했다. 실험설정변경은검증배치만,모델튜닝0/재균형0/특징변경0/새400일model0/제출0. 기존모델source/code/dataSHA와이번코드/계획SHA는매stage검사했다.', '', '원인추정신뢰도는낮음: LOOCV로학습자료를더남겼을때예측이어떻게달라지는지볼수있지만, 자료가부족한지입력정보가부족한지모델설정이문제인지혼자서는분리하지못한다. .5에서놓친날도score.2이상인날이있을수있으며,보조임계값.2는사전고정한민감도보고만이다. 일반일의오탐손해를새로검증하지않은상태에서임계값을바꾸거나EC회귀gate로채택할수없다.', '', '계획비평: 실제읽는10DIAG예측CSV의hash핀누락을v2에서보강했다. 첫2일fit후bootstrap에서빈F47pool float배열이연결index를float로승격시킨오류가났다. v3은연결index를int64로고정,원registration핀을유지하고continuation등록을추가했다. 모델·fit·저장예측은바꾸지않고중간집계만재실행했다. 중간은F13 120/121 단일block이라CI독립근거0. 최종비평내용은critique_final_v1.md에기록했다.', '', '## 재현·자료', '', '모든시드는8383/1919/7171. 학습은일sampleweight1/24·class_weight없음. 같은온실당일0..h RAW14/DP1/낮밤대비만features,정답·미래·다른온실현재입력제외. MASK불가5열은기존분류기와같이제외했다. 공식train_y값추가읽기0·소비40성능재점수0·공식test추론0·이전대회자료0.', '', 'PLAN_v2.md,run_v2.py prepare/first(첫2fit보존),bootstrap_repair_v1.md,run_v3.py score_midpoint/rest,registration_v1.json/continuation_registration_v1.json,local/predictions/26CSV+receipt,final_score_v1.json/local/final_paired_h15_v1.csv,training_exposure_v1.json,fresh_scalar_checks_v1.json,critique_*_v1.md 및critic_*recheck*가근거다. 재실행은기존폴더를덮어쓰지말고새버전폴더로등록해야한다.']
p=H/'report_v1.md';assert not p.exists();p.write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(dict(status='PASS',fresh=fresh,paired=x),ensure_ascii=True),flush=True)
