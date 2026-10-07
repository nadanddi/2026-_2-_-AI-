"""Persist actual score/verification progress, keeping old checkpoints intact."""
from pathlib import Path
import json,hashlib,re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['DOMAIN24_execution_assemble_v1.json','DOMAIN24_execution_verify_v1.json',
       'DOMAIN24_execution_score_v1.json','DOMAIN24_verified_gate_v1.json',
       'DOMAIN24_BLK_diagnostic_results_v1.json','DOMAIN24_score_independent_crosscheck_v2.json',
       'DOMAIN24_BLK_진단보고서_v1.md','feature_candidates_v10.csv',
       'critique_DOMAIN24_actual_gate_v1.md','critique_DOMAIN24_original_raw_plan_v1.md',
       'run_domain_original_raw_v2.py','crosscheck_original_preparation_v1.py',
       'DOMAIN24_original_matrix_reconstruction_probe_v1.json']
assert all((HERE/n).exists() for n in names)
for stage in ['assemble','verify','score']:
    r=json.loads((HERE/f'DOMAIN24_execution_{stage}_v1.json').read_text(encoding='utf-8'))
    assert r['returncode']==0 and r['frozen_sources_unchanged'] is True
result=json.loads((HERE/'DOMAIN24_BLK_diagnostic_results_v1.json').read_text(encoding='utf-8'))
assert len(result['results'])==48 and not any(r['BLK_screen_pass'] for r in result['results'])
folder=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2'
prep={p.name:sha(p) for p in folder.glob('*_fold*.json')}
record={'checkpoint':18,'domain_score_actual_exit0':True,'independent_Decimal_actual_exit0':True,
        'domain_variants':48,'domain_screen_passes':0,'domain_alpha':.025/54,
        'original_preparation_handle':77114,'handle_last_poll':'running; P2LOO_fold12 PASS',
        'preparation_receipts_snapshot':prep,'preparation_complete_exists':(folder/'complete.json').exists(),
        'files_sha256':{n:sha(HERE/n) for n in names},
        'raw_original_fit_registered':False,'raw_original_model_fit':False,
        'adoption':False,'submission':False,'whole_goal_complete':False,
        'next':['Poll original77114, never restart from lock/files alone',
                'Review scored diagnostics and rawrunner2 independently',
                'When all66 finish execute independent preparation crosscheck',
                'Fix original rawfit statistics/runtime/model contract before5346fits',
                'All24 original TM/P2LOO/EL1 x3seeds regardless BLK rank',
                'Literature24, data142/full subset and interaction grammar, firstunused confirmation once']}
with (HERE/'checkpoint_record_v18.json').open('x',encoding='utf-8') as h:json.dump(record,h,ensure_ascii=False,indent=2)
text=f'''# 現在 재개 지점 — 2026-10-07 집 코덱스 (checkpoint v18)

- 도메인 원조립55500/검증93729 및 등록score actual exit0. 전이180pins actualgate 독립비평 coreblocker0. 518400scalar max4.44e-16/22464source-future검사. Decimal checker2 actual exit0/48안864셀1728RMSE·allpCI 재검산PASS.
- domain48안 모두BLK FAIL alpha.025/54 유지. 보고서v1 및CSVv10 저장; 채택0. BLK순위로원24를줄이지않음. 고EC3일72행·작은8블록표본/CPUcachedbaseline 한계 유지.
- 원66prep handle77114 마지막직접poll running/P2LOO_fold12 PASS; 파일snapshot {len(prep)}/66, complete존재 {record['preparation_complete_exists']}. 과거 EL1_fold65는전체66완료표시가아님(실행순서DIAG→EL→P2). 원handle 계속확인·중복시작금지.
- firstoriginalfold actualv1 probe 전24freshmatrixSHA 일치. production materializer2는complete66와selfsource핀을강제. 원rawrunner1독립OR01/OR02→새runner2 resumeimputer/audit/최상위complete 보완. 아직fit등록/모델0; OR03 통계·후처리·runtime 계약봉인 남음. independent fullprep checker 작성/미실행.
- 다음: 원77114→66complete 독립검사; 도메인사후비평; rawrunner2/최종계약·통계실행전봉인;原24×3seed 66fold R3+cachedPFN/full후처리 검증. 문헌24/자료142·원열subset16383/interaction/미사용seed-layout1회/최종전체보고서 남음. goal active/GPU·제출0.

이하 이전 상태는 각 시점 기록이다.

'''
# Correct accidental non-Korean heading before writing.
text=text.replace('現在','현재').replace('原24','원24')
progress=HERE/'PROGRESS.md'
progress.write_text(text+progress.read_text(encoding='utf-8'),encoding='utf-8')
shared='''
### 2026-10-07 집 코덱스 도메인 BLK48 실제 진단 완료 (checkpoint v18)
- 등록assemble/verify/score actualexit0·source불변. actualgate독립180핀/source-future22464/scalar518400 max4.44e-16; Decimalchecker2 actualexit0/48안864cell1728RMSE/allpCI PASS. 도메인48전부screenFAIL alpha.025/54,새채택0·BLK순위로원24제외금지. DOMAIN24_BLK_진단보고서_v1.md/CSVv10 저장.
- 원77114 직접poll running/P2LOO_fold12 PASS,66준비 아직미완료. EL1fold65는fold번호이며전체완료아님. firstfold전24freshSHAprobe actualPASS. 원rawrunner2 OR01/02 resume보완·fit등록0; source비평/통계/runtime계약 및complete66검산 남음. 전체원검증기/문헌/자료/최종firstunused1회/전체보고서 미완료. goal active/GPU·제출0.
'''
for p in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-07.md']:
    with p.open('a',encoding='utf-8') as h:h.write(shared)
catalog=ROOT/'공용/데이터_단서_카탈로그.md'
old=catalog.read_text(encoding='utf-8')
nums=[int(n) for n in re.findall(r'\| 6\.(\d+) \|',old)]
assert max(nums)==381, 'Re-read concurrent catalog before appending'
entry='''
| 6.382 | **도메인24 BLK48 실제 전체 진단FAIL·원66 준비 계속** (2026-10-07 집 코덱스) | 6.376~378등록 raw72+3replay→assemble/verify/score actualexit0/source불변,actualgate독립180핀/source-future22464/scalar518400 max4.44e-16. Decimalchecker2 actualexit0/48안864셀1728RMSE·블록SSE·allpCI 대조PASS. 누적54 alpha.025/54 사전기준에서48안통과0;새채택0·고EC3일72행/8block노출탐색·과거GPU제출동등성미주장. 원24×3seed TM/P2LOO/EL1은FAIL과무관하게전부필수. 원77114 running/P2LOO12까지directpoll;原66완료아님. firstfold24freshmatrixSHA actualPASS;rawrunnerOR01/02새v2보완·model-fit등록0. 문헌/자료/최초미사용확인/전체보고서미완료·goal active/GPU·제출0 | 집/코덱스/analysis/ec_feature_rebuild_20261007_v1/DOMAIN24_BLK_진단보고서_v1.md;DOMAIN24_BLK_diagnostic_results_v1.json;DOMAIN24_score_independent_crosscheck_v2.json;critique_DOMAIN24_actual_gate_v1.md;checkpoint_record_v18.json |
'''.replace('原66','원66')
with catalog.open('a',encoding='utf-8') as h:h.write(entry)
print(f'checkpoint18/catalog6.382 persisted; prep snapshot {len(prep)}/66, goal incomplete')
