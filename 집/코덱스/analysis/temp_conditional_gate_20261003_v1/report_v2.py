from pathlib import Path
import json,csv,collections,math,statistics
H=Path(__file__).resolve().parent;O=H.parents[1]/'local/temp_conditional_gate_20261003_v1'
v=json.loads((H/'verification.json').read_text(encoding='utf-8'));r=json.loads((H/'result.json').read_text(encoding='utf-8'));a=json.loads((H/'gate_audit.json').read_text(encoding='utf-8'));re=json.loads((H/'replay_verification.json').read_text(encoding='utf-8'))
assert v['status']==a['status']==re['status']=='PASS' and v['verdicts']['G1']=='REJECT'
lines=['# 온도 조건별 혼합비중 G1 · 2026-10-03','', '사용자 제안은 가능하다. 입력의 조건마다 전문가 모델의 비중을 바꾸는 gating model/mixture of experts다. 현재W30G도현재온도만으로저온보호gate를쓰고있다. 이번에는온도외시간대·변화·운전조건을추가로학습하는G1을구현했다. 이 구체 G1은 외부 검증 실패로 기각하며 기존 W30G 유지.','', '## 이번 적용 범위와 식','', '- 원BASE/CODEX/TabPFN 모델의예측은동일. 바꾸는것은BASE/CODEX상대비중,TabPFN .3g와≤8℃보호는고정. PFN비중까지새로학습한3전문가gate라고표현하지않는다.','- gate조건: farm_id,sin,cos,현재온도,당일온도평균/표준편차/차분,내외온도차,일사,환기,난방,습도. 모두예측행현재·이전입력만.','- 내부3fold에서원BASE전체3모델(.65물리+LGB/.25Ridge/.10Nys)와CODEX를새로학습한OOF로비중회귀를적합했다. 외부query정답을다른fold모델이보는기존글로벌OOF메타학습은쓰지않았다. 농장5일묶음순환·±1일buffer·두시드, 총72내부전문가적합묶음.','- Z=[1,표준화조건], beta0=(.4+.1g)/(1−.3g), Ridge100(no-intercept)의 design=Z*g*(BASE−CODEX),target=y−beta0*BASE−(1−beta0)*CODEX. delta=g*clip(Zθ,−.15,.15).','- wBASE=.4+.1g+(1−.3g)delta, wCODEX=.6−.4g−(1−.3g)delta,wPFN=.3g. 비음수/합1,≤8℃delta0. 최종예측은세원모델예측의가중합. effective CODEX는기존검산runner호환표현일뿐새CODEX모델아님.','- gate의학습목표는PFN제외B/C normalized pool의RMSE다. PFN오차와의상호작용까지맞춘최적화는아니므로전체혼합의외부성능으로판정해야한다.','', '## 원 W30G 대비 외부 결과','', 'DIAG10 400일/9,600행, EXT10 85일/2,040행, EXT12 219일/5,256행. 두BASE시드7/101(CODEX726/727)×PFN문맥1~8/17~24×3검증기=12칸. 양수는RMSE악화.','', '| 검증기 | RMSE 변화 |','|---|---:|']
for val in ['DIAG10','EXT10','EXT12']:
    p=[s['delta_pct'] for s in v['cells'] if s['validator']==val];lines.append(f'| {val} | {min(p):+.3f}~{max(p):+.3f}% |')
improved=sum(s['delta_pct']<0 for s in v['cells']);ps=[s['p_worse'] for s in v['cells'] if s['validator']=='DIAG10'];lines+=['', f'개선 방향 {improved}/12칸,DIAG p_worse {min(ps):.5f}~{max(ps):.5f}. 사전main8b10e32,이전15+이번1=16안 보수적 보정 p<.025/16·99.6875% ΔMSE CI상한<0및전12칸개선기준실패. 이고정G1기각판정신뢰도높음. 조건별비중학습전체가불가능하다는증거는아니다.', '', '## 실제 행별 비중','', '| 시드 | 조건 | n | BASE 평균/범위 | CODEX 평균/범위 | PFN 평균/범위 |','|---|---|---:|---:|---:|---:|']
for s in a['weight_segments']:
    if s['segment'] not in ['all','cold','night','day','evening']:continue
    cells=[f'{s[f"w_{name}_mean"]:.3f} / {s[f"w_{name}_min"]:.3f}~{s[f"w_{name}_max"]:.3f}' for name in ['BASE','CODEX','PFN']];lines.append(f'| {s["seed"]} | {s["segment"]} | {s["n"]} | '+' | '.join(cells)+' |')
lines+=['', '이는DIAG검증행의실제학습비중이며누가더잘맞는다는인과증거가아니다. cold는현재온도≤8℃,밤00~05/낮06~17/저녁18~23. cold기존.4/.6/0정확유지.','', '## 학습 점수·오차 위치와 검산','']
st=[s['scores']['G1'] for s in r['training_scores'] if s['validator']=='DIAG10'];lines.append(f'- DIAG 내부B/C pool RMSE비중적합전범위 {min(s["inner_pool_before_rmse"] for s in st):.4f}~{max(s["inner_pool_before_rmse"] for s in st):.4f},적합후 {min(s["inner_pool_after_rmse"] for s in st):.4f}~{max(s["inner_pool_after_rmse"] for s in st):.4f}. 게이트자체는여기정답으로적합했으므로훈련점수이지독립성능증거아님. 전체W30G점수와구별한다.')
for kind in ['level','shape']:
    s=[s for s in v['segments'] if s['segment']=='level_shape'];p=[100*(s[f'candidate_{kind}_rmse']/s[f'baseline_{kind}_rmse']-1) for s in s];lines.append(f'- DIAG {kind} RMSE변화 {min(p):+.3f}~{max(p):+.3f}%.')
for seg in ['F13','F47','early','late','hour00_05','hour06_17','hour18_23']:
    p=[s['delta_pct'] for s in v['segments'] if s['segment']==seg];lines.append(f'- {seg} 변화 {min(p):+.3f}~{max(p):+.3f}%.')
with (O/'oof.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
stats=[]
for seed in ['726','727']:
    for ctx in ['1-8','17-24']:
        scores=[];delta=[]
        for k in range(10):
            d=[x for x in rows if x['validator']=='DIAG10' and x['seed']==seed and x['context']==ctx and int(x['fold'])==k];b=math.sqrt(math.fsum((float(x['base'])-float(x['sub_temp']))**2 for x in d)/len(d));c=math.sqrt(math.fsum((float(x['candidate'])-float(x['sub_temp']))**2 for x in d)/len(d));scores.append(c);delta.append(c-b)
        stats.append(dict(seed=seed,context=ctx,fold_sd=statistics.stdev(scores),paired_delta_sd=statistics.stdev(delta),improved_folds=sum(x<0 for x in delta)))
lines.append(f'- fold RMSE표준편차 {min(s["fold_sd"] for s in stats):.4f}~{max(s["fold_sd"] for s in stats):.4f},paired delta표준편차 {min(s["paired_delta_sd"] for s in stats):.4f}~{max(s["paired_delta_sd"] for s in stats):.4f}.')
lines+=['', '- raw train_y/CSV math.fsum RMSE/혼합식/20k농장5일블록bootstrap p/CI 독립일치 PASS. 외부학습·검증일±1buffer/내부학습subset/query전체합을독립집합으로재확인.','- raw인과features에서median/scaler와design/target재구성,ΣXᵀWX+100I닫힌식으로계수독립계산,δclip과세원모델가중합·비중합1·비음수·≤8℃불변을전24조건재검산 PASS.','- raw MASK부터첫outer DIAGfold726의모든내부BASE/CODEX를재학습해예측/비중/design 최대차0. 모든외부fold반복적합은하지않음.','- 새전처리/모델은innertrain에서만fit,gate전처리는outertrain에서만fit. 기존TF global입력rank는고정비교조건유지한계. BASE/PFN외부cache는원레시피재사용. PFN두문맥은gate학습에쓰지않고외부검증에만썼다.','- 원CV반복사용·별도미사용홀드아웃/리더보드검증아님. 샘플링없는CODEX726/727은실질독립시드아니지만BASE7/101변동검사. 가중치학습정답사용은오직outertrain일이며평가query정답경로없음.','- 반론:조건별비중이라성능이반드시올라야하는가? 내부조건별모델오차관계가다른날에도유지되고다른전문가와의오차상관에맞아야한다. 작은메타학습표본/inner모델과outer모델학습크기차이/PFN을고정한학습목표등때문에추가gate도일반화실패가능. 각원인의기여는분리하지않았다.','- 원검증불통과라전체W30G날씨GUARD추가채택검증안함. 전worker종료·test평가예측/제출파일/업로드/EC잠금0. 큰결과local/temp_conditional_gate_20261003_v1,source/검산내analysis.','']
(H/'결과보고서_v1.md').write_text('\n'.join(lines),encoding='utf-8');(H/'fold_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print('\n'.join(lines[16:23]))
