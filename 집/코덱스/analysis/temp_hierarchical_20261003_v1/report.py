from pathlib import Path
import json,statistics,math
H=Path(__file__).resolve().parent
r=json.loads((H/'result.json').read_text(encoding='utf-8'));v=json.loads((H/'verification.json').read_text(encoding='utf-8'));replay=json.loads((H/'replay_verification.json').read_text(encoding='utf-8'))
lines=['# 온도 하루 수준·시간 모양 분리 실험 결과 · 2026-10-03','', '사용자 “더 해봐”에 따라 온도 타깃 구조를 바꾼 두 안을 실제 실행했다. 두 안 모두 사전 기준 불통과. 기존 7회차 W30G 유지. 제출 후보 확정이나 제출 파일 생성은 하지 않았다.','', '## 변경과 검증','', '- 원래 CODEX 물리 Ridge를 유지하고 잔차를 학습일 평균(level)과 평균을 뺀 시간 변화(shape)로 나눴다. level은 하루 1행의 0시 센서로, shape는 현재까지 입력으로 학습했다. H1은 level LGB, H2는 level Ridge이며 shape LGB와 W30G 혼합 비중은 고정했다. 학습일 평균 정답은 학습 타깃으로만 썼다.','- 사전 커밋 780926c, 실행 보정 f26699e. run.py의 day 열 누락은 검증 예측 생성 전 실패했고 run_v2.py에서 drop=False로만 수정했다. 검산 verify.py의 온실 범위 오류는 verify_v2.py에서 날씨 검사 대상 F13/F47로 제한했다. 모델·기준·예측은 바꾸지 않았다.','- DIAG10 400일/9,600행, EXT10 85일/2,040행, EXT12 219일/5,256행. 후보마다 2시드×2PFN 문맥×3검증기의 12칸. 이번 세션 전체 7가설로 보수적 본페로니 alpha=.025/7, 온실별5일 블록20,000회. all directions 및 DIAG p_worse/CI 기준 고정.','', '| 안 | DIAG10 RMSE 변화 | EXT10 변화 | EXT12 변화 | 개선 칸 | DIAG p_worse | 판정 |','|---|---:|---:|---:|---:|---:|---|']
for tag in ['H1','H2']:
    ss=[x for x in v['cells'] if x['scope']=='W30G' and x['member']==tag]
    parts=[]
    for val in ['DIAG10','EXT10','EXT12']:
        a=[s['delta_pct'] for s in ss if s['validator']==val];parts.append(f'{min(a):+.3f}~{max(a):+.3f}%')
    ps=[s['p_worse'] for s in ss if s['validator']=='DIAG10'];lines.append(f'| {tag} | '+ ' | '.join(parts)+f' | {sum(s["delta_pct"]<0 for s in ss)}/12 | {min(ps):.5f}~{max(ps):.5f} | 기각 |')
lines+=['','양수는 RMSE 악화다. H2의 EXT10 국소 개선은 전체 채택 조건을 만족하지 않는다. H1/H2 모두 저온 EXT12에서 악화했으므로 이 결과를 보고 저온만 다른 비중으로 튜닝하지 않았다. 신뢰도: 이번 고정 적용안의 불통과 판정은 높음. 리더보드 성능이나 다른 level/shape 방식의 불가능성은 판단하지 않음.','', '## 오차 분해와 학습 격차','']
for tag in ['H1','H2']:
    seg=[s for s in v['segments'] if s['scope']=='W30G' and s['member']==tag and s['segment']=='level_shape']
    for kind in ['level','shape']:
        pct=[100*(s[f'candidate_{kind}_rmse']/s[f'baseline_{kind}_rmse']-1) for s in seg];lines.append(f'- {tag} DIAG {kind} RMSE 변화 {min(pct):+.3f}~{max(pct):+.3f}%.')
    st=[s['scores'][tag] for s in r['training_scores'] if s['validator']=='DIAG10'];lines.append(f'- {tag} 원 DIAG 구성원 fold 학습 RMSE 범위 {min(s["train_rmse"] for s in st):.4f}~{max(s["train_rmse"] for s in st):.4f}, 검증 {min(s["validation_rmse"] for s in st):.4f}~{max(s["validation_rmse"] for s in st):.4f}. 전체 W30G 학습 점수와 구별한다.')
    for segname in ['F13','F47','early','late','hour00_05','hour06_17','hour18_23']:
        a=[s['delta_pct'] for s in v['segments'] if s['scope']=='W30G' and s['member']==tag and s['segment']==segname];lines.append(f'- {tag} {segname}: {min(a):+.3f}~{max(a):+.3f}%.')
lines+=['', '## 날씨 그룹 의존성 진단','', '- train_X의 날씨 4센서×24시간을 기준으로 유사 거리≤.05 연결 성분을 만들었다. 400일/122그룹/491유사 쌍, 날씨가 정확히 같은 438쌍. 이 유사 쌍에서 전체14센서 입력 완전복사와 온도 정답 완전복사는 각각0. 원 DIAG 검증일291/400일이 학습일과 날씨 그룹을 공유했다. CSV 벡터 해싱과 독립 버퍼 집합으로 438쌍·291일 재확인.','- GUARD는 ±1일 버퍼 후 검증 날씨 그룹의 학습일을 추가로 제외했다. 추가 제거는 fold별26~48일. 새 H1/H2, 기존 CODEX/CB/ET 구성원을 같은 조건으로 재학습했다. 이 점수에는 BASE/PFN이 없어 W30G 전체 성능으로 표현할 수 없다. 날씨 공유 자체를 정답 누수라고 해석하지 않는다.','', '| 구성원 | 원 DIAG 대비 GUARD RMSE 변화 |','|---|---:|']
for tag in ['CODEX','CB','ET']:
    a=[s['delta_pct'] for s in v['prior_member_guard_comparison'] if s['member']==tag];lines.append(f'| {tag} | {min(a):+.3f}~{max(a):+.3f}% |')
lines+=['', 'GUARD 기준 CODEX보다 H1은 +2.229~+2.383%, H2는 +20.535% 악화했다. H2의 0시 선형 수준 예측은 이 분할에 특히 민감했다. 날씨 그룹 제외는 학습일도 줄이므로, 일반화 격차를 전부 날씨 공유 때문이라고 인과적으로 단정할 수 없다. 신뢰도: 조건부 점수 비교 높음, 원인 해석 낮음.','', '## 검증과 한계','', '- 원시 train_y와 모든 OOF 정답 일치, 행 중복 없음. CSV math.fsum RMSE와 numpy RMSE, 독립 혼합 식, 모든 블록 bootstrap p/CI, 추가 제외 train/validation 집합 일치: verification.json PASS. 원시 MASK 특징부터 첫 DIAG fold seed726을 재학습해 H1/H2/physics 예측 차0: replay_verification.json PASS. 모든 fold 재학습 반복은 하지 않았다.','- 새 imputer/scaler/physics/level/shape는 fold 안에서만 fit. 0시값은 같은 온실/날의 0시 관측부터 forward-fill, 다른 특징은 expanding/현재·과거 입력이다. 원 TF 입력 기반 가중치의 전체 학습일 rank는 비교 기준 그대로 유지했으며 fold 내 재정의하지 않았다.','- H2/shape/물리식은 샘플링 없이 시드 예측이 같을 수 있어 시드 수를 독립 증거로 세지 않는다. PFN 문맥 1~8,17~24와 BASE7/101은 저장 OOF 재사용. 반복해서 본 CV이며 독립 미사용 홀드아웃/최종3시드 제출모델/리더보드로 검증한 결과가 아니다.','- 새 후보는 기준 불통과로 기각하므로 전체 W30G 날씨 GUARD 추가 채택 검증은 필요하지 않다. 개선 가능성 전체를 기각한 결과는 아니다.','- 모든 학습 worker 종료, test 평가 예측·제출 파일·플랫폼 업로드·EC 최종 잠금 열람0. 큰 fold 결과는 집/코덱스/local/temp_hierarchical_20261003_v1에 둔다.','']
(H/'결과보고서_v1.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines[12:19]));print('Report written')
