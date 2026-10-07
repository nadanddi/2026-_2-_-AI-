from pathlib import Path
import json,re,datetime
here=Path(__file__).resolve().parent;root=here.parents[3]
catalog=root/'공용/데이터_단서_카탈로그.md'
last=max(int(n) for n in re.findall(r'\| 6\.(\d+) \|',catalog.read_text(encoding='utf-8')))
assert last==379,last
synthetic=json.loads((here/'CH2_prefix_synthetic_audit_v2.json').read_text(encoding='utf-8'))
replay=json.loads((here/'CH2_training_reference_file_replay_v1.json').read_text(encoding='utf-8'))
assert len(synthetic['checks'])==synthetic['check_count']==48
note='''
### 2026-10-07 집 코덱스 CH2 경계 보완·학습 참조 실제 준비 (checkpoint v16)
- 직전 goal turn은progress(PF실제메타데이터·누락정정). 이번도progress: prefix3에서 same-block 과거 모든24h/current0..h exact packet과 전체ID/schema검사를 수치변환 전에 강제. 다른block·중간hour·이전day 누락은 거부. synthetic v2 actual48검사 PASS; 독립critic CP01/CP02 닫힘/새packetblocker0.
- prepare_ch2_training_reference_v1 실제exit0: 허용train5520행230기록/원201links/17cyclecomponent, 공개학습숫자만 변환. 원graph·train-onlyscale 보존·immutable JSON roundtrip exact. 독립준비비평 current8SHA/파일SHA일치·핵심blocker0.
- ch2_reference_loader_v1 실제exit0: 저장파일 재읽기 SHA/canonicaldigest/전이적pin/import경로/scale/records replay 검사. 별도 실제 -O 실행은 파일읽기 전 RuntimeError exit1(의도한 negative PASS). PowerShell independent check48/파일SHA일치·5520행/query numeric0. loader비평요청중.
- 실제query예측/학습/후처리/효과채점/3방법사전등록은 아직0. 다음은 정확3방법×2scope/기존6+도메인48+새6 비교누적60 계획·runner 및gate를 실제코드로 봉인 후비평. 기존도메인alpha54/등록은수정하지않음. 원검증기는fold별graph/scale재생성 필수·전체원24시험필수.
- 도메인assemble 승인credits오류 미실행/복구확인없음/우회0. 원TM/P2LOO/EL1×3seed·CH2실효과·문헌24·데이터142·최초미사용seed-layout1회·최종보고서 모두남음. goal active/GPU·채택·제출0. 카탈로그6.380.
'''
record={'checkpoint_version':16,'recorded_at':datetime.datetime.now().astimezone().isoformat(),
        'previous_goal_turn':'progress: omission correction and actual PF metadata',
        'this_goal_turn':'progress: actual prefix guards48 and train reference/fresh file replay',
        'synthetic_checks':48,'train_rows':replay['train_rows'],'train_records':replay['train_records'],
        'query_inputs_numerically_parsed':0,'candidate_execution_registered':False,
        'CH2_performance_evaluated':False,'Python_O_negative_test':'actual exit1 RuntimeError before file load',
        'loader_critic_pending':True,'assembly_status':'not executed; workspace credits approval failure unresolved/no bypass',
        'goal_scope_complete':False,'GPU_used':False,'submission_created':False}
out=here/'checkpoint_record_v16.json';assert not out.exists()
out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
progress=here/'PROGRESS.md'
progress.write_text('# 現在 재개 지점 — 2026-10-07 집 코덱스 (checkpoint v16)\n'+note+'\n이하 과거 상태는 시점별 스냅샷.\n\n'+progress.read_text(encoding='utf-8'),encoding='utf-8')
for path in [root/'공용/HANDOFF.md',root/'집/코덱스/작업일지/2026-10-07.md']:
    with path.open('a',encoding='utf-8') as handle:handle.write(note)
with catalog.open('a',encoding='utf-8') as handle:
    handle.write('\n| 6.380 | **CH2 same-block 완전 prefix 검사 보완·학습 reference 파일 실제 재생 PASS** (2026-10-07 집 코덱스) | 6.379 후 CP01다른block수치접근/CP02불완전prefix 허용을 prefix3로보완: 전체key/schema먼저검사,이전querydays24h+현재0..h exact강제. synthetic actual48PASS/독립비평문제닫힘. training-only5520행230records/201links/17cycliccomponent 저장·JSON재생,준비critic현재8pins일치. loader actual파일SHA/canonicaldigest/전이핀/import경로/scale/replay PASS,실-O는fileload전RuntimeError exit1/의도한negative,PS48및SHA일치. query/gap/heldout숫자0·실제inference/score/후속3후보등록0. 물리source/성능증거아님·loadercriticpending. 도메인assemble credits승인실패 미실행/우회0·전체goal미완료/GPU·채택·제출0 | 집/코덱스/analysis/ec_feature_rebuild_20261007_v1/CH2_prefix_synthetic_audit_v2.json;CH2_training_reference_preparation_v1.json;CH2_training_reference_file_replay_v1.json;critique_CH2_prefix_sources_v2.md;critique_CH2_training_reference_v1.md;checkpoint_record_v16.json |\n')
print('Recorded checkpoint16 and catalog6.380; full goal active')
