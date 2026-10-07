from pathlib import Path
import json,datetime,re

here=Path(__file__).resolve().parent
root=here.parents[3]
pf=json.loads((here/'CLAUDE_PF_outputs_metadata_v2.json').read_text(encoding='utf-8'))
catalog=root/'공용/데이터_단서_카탈로그.md'
text=catalog.read_text(encoding='utf-8')
last=max(int(n) for n in re.findall(r'\| 6\.(\d+) \|',text))
assert last==378, last
record={'checkpoint_version':15,'recorded_at':datetime.datetime.now().astimezone().isoformat(),
 'user_request':'누락 재점검; 바닐라 데이터 범위와 전체 실험 목표 구분',
 'assembly_status':'not executed; automatic approval review workspace credits failure unresolved; no bypass',
 'CH2_assignment_audit':'actual stdlib exit0 independent optimality audit; physical source or predictive validity not proved',
 'CH2_prefix_audit':'36 synthetic checks; no real-query prediction/model fit/score; source critic requested',
 'PF_outputs':[{k:v for k,v in d.items() if k!='missing_expected_keys'} for d in pf['outputs']],
 'PF_metadata_limits':'not causal gate or completed-model receipt; running process unknown; no target scoring',
 'vanilla_scope':'raw14, row_id, same-farm current/past inputs, public training input/EC/temp including EC-unlabeled farms; exclude external/derived values',
 'goal_remaining':'DOMAIN assemble/gate/score; original all24 x3seeds TM/P2LOO/EL1; CH2; literature24/data142; first-unused seed/layout once; final report',
 'whole_goal_complete':False,'GPU_used':False,'submission_created':False}
out=here/'checkpoint_record_v15.json'
assert not out.exists()
out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
note='''
### 2026-10-07 집 코덱스 누락 재점검 (checkpoint v15)
- 원자료 목록의 EC 무정답 온실 train_X·공개 온도 정답·row_id·같은 온실 과거 입력·공개 학습 이웃 원입력/정답을 재확인. 이웃 거리/보간/예측은 파생이며 바닐라 원천 목록과 구분한다. sync_start는 앞선 세션에서 실제 완료; 재실행하지 않음.
- CH2 학습 cost/assignment 독립 stdlib audit actual exit0: 201 links/46 components/17 cycles. 이는 정답 기반 비용의 최적성 확인이며 실제 출처·예측 유효성 증거 아님. prefix v2 synthetic36 PASS; 실제 query/model/score0, 독립 비평 요청.
- ignored local 경로 재검색으로 PF1/PF2 CSV 발견. v2 metadata actual exit0: PF1 10848행66fold 전체키 일치; PF2 6648행34fold/4200행 미포함. 공통6648키 공유예측 최대4.44e-16. 파일 존재를 완료/live/causal gate로 해석하지 않음. v1 full assertion 실패 보존; target 숫자 변환·재채점0.
- DOMAIN assemble 자동승인 credits 실패 미실행 상태 유지/승인우회0. 원검증기·CH2·문헌24·데이터142·미사용seed-layout1회·전체보고서 남음; goal active/GPU·새채택·제출0. 카탈로그6.379.
'''
for path in [root/'집/코덱스/작업일지/2026-10-07.md',root/'공용/HANDOFF.md']:
    with path.open('a',encoding='utf-8') as handle:handle.write(note)
progress=here/'PROGRESS.md'
old=progress.read_text(encoding='utf-8')
progress.write_text('# 현재 재개 지점 — 2026-10-07 집 코덱스 (checkpoint v15)\n'+note+'\n이하 과거 상태는 해당 시점 스냅샷이다.\n\n'+old,encoding='utf-8')
with catalog.open('a',encoding='utf-8') as handle:
    handle.write('\n| 6.379 | **이웃 원천과 바닐라 범위 재확인·PF 파일 누락 정정·CH2 독립/합성 감사** (2026-10-07 집 코덱스) | ignored local 재검색: PF1 10848행66fold 전체키, PF2 6648행34fold 부분/4200행 미포함, 공유6648 최대4.44e-16; metadata만/정답숫자변환0·성능/causal gate 미판정. CH2 독립201link 비용/할당 최적성 actual PASS·46component/17cycle, 실제 출처 증거 아님. prefix v2 synthetic36 PASS·실데이터fit/score0·critic요청. 바닐라14원입력/row_id/같은온실과거/공개학습이웃/EC무정답온실·공개온도 포함, 거리·보간·예측은 파생 제외. assemble credits 승인오류 미실행/우회0·전체goal active·새채택/GPU/제출0 | 집/코덱스/analysis/ec_feature_rebuild_20261007_v1/CLAUDE_PF_outputs_metadata_v2.json;CH2_reference_independent_audit_v1.json;CH2_prefix_synthetic_audit_v1.json;checkpoint_record_v15.json |\n')
print('Recorded checkpoint15, PROGRESS, HANDOFF, worklog, catalog6.379')
