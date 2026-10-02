from pathlib import Path
import json,csv,math,statistics,collections
HERE=Path(__file__).resolve().parent;F=HERE.parent/'temp_farm_physics_20261003_v1'
lines=['# 온도 추가 탐색4안 · 2026-10-03','', '사용자 추가 탐색 지시에 따라 타깃을 나누지 않는 손실3안과 물리기울기1안을 사전 고정 후 실행했다. 모두 원 W30G의 CODEX 구성원만 교체했다. 원래 혼합 비중·게이트·BASE·PFN은 동일하다. 4안 모두 채택 기준 불통과, 기존 W30G 유지.','', '## 결과','', '각 안은 DIAG10 400일/9600행, EXT10 85일/2040행, EXT12 219일/5256행. 2시드×2PFN 문맥×3검증기, 후보당12칸. 아래 양수는 RMSE 악화다.','', '| 안 | 바꾼 부분 | DIAG10 | EXT10 | EXT12 | 개선 칸 | DIAG p_worse |','|---|---|---:|---:|---:|---:|---:|']
descriptions={'L1':'하루 수준 오차의 손실2배','S1':'하루 수준 손실½, 시간 모양 유지','D1':'같은날 연속시간 증분 오차 추가','F1':'두기록 물리18열 기울기 분리'}
datasets=[]
for directory,tags,out in [(HERE,['L1','S1','D1'],'temp_group_loss_20261003_v2'),(F,['F1'],'temp_farm_physics_20261003_v1')]:
    v=json.loads((directory/'verification.json').read_text(encoding='utf-8'));r=json.loads((directory/'result.json').read_text(encoding='utf-8'));replay=json.loads((directory/'replay_verification.json').read_text(encoding='utf-8'));assert v['status']==replay['status']=='PASS' and all(q=='REJECT' for q in v['verdicts'].values());datasets.append((directory,tags,v,r,out))
    for tag in tags:
        ss=[x for x in v['cells'] if x['member']==tag];parts=[]
        for val in ['DIAG10','EXT10','EXT12']:
            a=[x['delta_pct'] for x in ss if x['validator']==val];parts.append(f'{min(a):+.3f}~{max(a):+.3f}%')
        p=[x['p_worse'] for x in ss if x['validator']=='DIAG10'];lines.append(f'| {tag} | {descriptions[tag]} | '+' | '.join(parts)+f' | {sum(s["delta_pct"]<0 for s in ss)}/12 | {min(p):.5f}~{max(p):.5f} |')
lines+=['','L1/F1 일반검증의 작은 감소는 통계 기준 미달이고 EXT12에서 방향이 반대다. S1/D1은 일반검증에서도 악화했다. 이 네 구체안의 불통과 판정 신뢰도는 높지만, 모든 손실·물리식 개선이 불가능하다는 증거는 아니다.','', '## 사전 고정과 구현 확인','', '- L1/S1/D1 main aeafd7c에서 등록. 전체 학습일의 정답은 손실 계산에만 쓰며 평가 시점의 특징은 현재·과거 자기 온실 입력만. 그룹 수준 항은 Σw e² + λΣday Wday mean_w(e)²(λ1,−.5), D1은 같은날 연속시각의 오차 증분 제곱을 추가한다. 가중치는 gradient/hessian에 직접 한 번 적용했다. custom Hessian의 비대각 항은 LGB에서 근사하므로 exact Newton 해법은 아니다.','- 손실 선택자가 wrapper의 sample_weight 인자로 오인된 v1은 세 안 모두 D1 실행이었고 전체 무효 처리했다. 실행 보정 de566be. 이어 동명 모듈 충돌을 적합 전에 고유 모듈명으로 고침(3e09e84). 최종 run_v3.py/group_loss_model_v2.py와 새 local/temp_group_loss_20261003_v2 결과만 이 보고서에 사용. 이전 v1 결과는 보존하되 판정 제외. 모델 수식·비중·채택 기준 변경0.','- 수치미분으로 gradient/Hessian 대각 최대차5.36e−11 이하. ZERO손실로 원 CODEX 재현 최대차1.93e−9, RMSE차1.46e−12. 실제 corrected L1/S1/D1 첫fold 예측은 서로 달라 선택자 동작 확인.','- F1 main3734559에서 별도 순차 등록. 물리19열에 farm_id 제외18열×(farm_id−.5)만 추가. median/scaler/Ridge100, 후속 LGB220/.035/leaf12/minchild100/L2=15 유지. F13/F47은 여러출처의 합성 기록이라 이 결과로 실제온실 열용량을 측정했다고 해석하지 않는다.','- 손실3안은 앞선7안 포함 보수 alpha=.025/10, F1은 .025/11. 모든12칸 감소 및각DIAG bootstrap CI상한<0/p_worse기준을 고정했다. 모두 더 느슨한 .025에서도 실패하므로 누적 보정 선택 때문에만 기각한 것이 아니다.','', '## 오차 분해·학습 격차·fold 변동','']
audit=[]
for directory,tags,v,r,out in datasets:
    raw=HERE.parents[1]/'local'/out/'oof.csv'
    with raw.open(encoding='utf-8',newline='') as f:records=list(csv.DictReader(f))
    for tag in tags:
        level=[s for s in v['segments'] if s['member']==tag and s['segment']=='level_shape']
        for kind in ['level','shape']:
            a=[100*(s[f'candidate_{kind}_rmse']/s[f'baseline_{kind}_rmse']-1) for s in level];lines.append(f'- {tag} DIAG {kind} RMSE 변화 {min(a):+.3f}~{max(a):+.3f}%.')
        st=[s['scores'][tag] for s in r['training_scores'] if s['validator']=='DIAG10'];lines.append(f'- {tag} 구성원 fold 학습RMSE 범위 {min(s["train_rmse"] for s in st):.4f}~{max(s["train_rmse"] for s in st):.4f}, 검증 {min(s["validation_rmse"] for s in st):.4f}~{max(s["validation_rmse"] for s in st):.4f}. 전체W30G 학습점수는 아니다.')
        for name in ['F13','F47','early','late','hour00_05','hour06_17','hour18_23']:
            a=[s['delta_pct'] for s in v['segments'] if s['member']==tag and s['segment']==name];lines.append(f'- {tag} {name} 변화 {min(a):+.3f}~{max(a):+.3f}%.')
        cells=collections.defaultdict(list)
        for x in records:
            if x['member']==tag and x['validator']=='DIAG10':cells[(x['seed'],x['context'],int(x['fold']))].append(x)
        sd=[]
        for seed in ['726','727']:
            for ctx in ['1-8','17-24']:
                scores=[];deltas=[]
                for k in range(10):
                    rows=cells[(seed,ctx,k)];a=math.sqrt(math.fsum((float(x['base'])-float(x['sub_temp']))**2 for x in rows)/len(rows));b=math.sqrt(math.fsum((float(x['candidate'])-float(x['sub_temp']))**2 for x in rows)/len(rows));scores.append(b);deltas.append(b-a)
                sd.append(dict(seed=seed,context=ctx,fold_mean=statistics.mean(scores),fold_sd=statistics.stdev(scores),paired_delta_sd=statistics.stdev(deltas),improved_folds=sum(x<0 for x in deltas)))
        audit.append(dict(member=tag,fold_statistics=sd));lines.append(f'- {tag} W30G 후보 DIAG fold RMSE 표준편차 {min(x["fold_sd"] for x in sd):.4f}~{max(x["fold_sd"] for x in sd):.4f}; paired RMSE차 표준편차 {min(x["paired_delta_sd"] for x in sd):.4f}~{max(x["paired_delta_sd"] for x in sd):.4f}.')
lines+=['','## 검산과 한계','', '- 네 안의 원시train_y 정답일치, 독립CSV math.fsum RMSE, 혼합식, 20k5일농장블록bootstrap p/CI 모두 PASS. 첫 원DIAG fold726을 raw MASK특징부터 다시 만들어 적합해 네 안/물리 예측 최대차0. 모든fold 재학습 반복은 하지 않았다.','- 검증/학습 전처리 분리: 새물리median/scaler/모델은fold학습만fit. 특징추가F1은현재입력변환만. D1은훈련목적함수에미래학습정답이포함되나 평가행의미래정답/미래입력으로예측하는경로는없음. 라벨이학습목적에쓰이는것과평가미래누수는구별한다.','- 기존TF가중치global 입력rank를그대로사용한조건부비교. BASE/PFN저장OOF를재사용. 샘플링없는 CODEX에서726/727의실제예측은같을수있으므로 독립시드증거2개로세지않음. BASE7/101와PFN문맥에따른전체혼합변동도함께검사했다.','- 같은CV를여러번본탐색이며독립미사용홀드아웃·리더보드·최종제출3시드 모델검증은아니다. 네 안모두원검증불통과라전체W30G 날씨GUARD 채택검증은실행하지않음.','- 진행중worker없음. test평가예측·제출파일·업로드·EC잠금열람0. 원소스/실험산출물수정없이새버전으로보존. 큰결과는 local/temp_group_loss_20261003_v2, local/temp_farm_physics_20261003_v1.','']
(HERE/'결과보고서_v1.md').write_text('\n'.join(lines),encoding='utf-8');(HERE/'fold_variance_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('\n'.join(lines[9:15]));print('Report written')
