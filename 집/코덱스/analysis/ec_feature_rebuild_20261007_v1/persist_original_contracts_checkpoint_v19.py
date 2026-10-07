"""Record concrete pre-fit progress without implying original-model validation."""
from pathlib import Path
import hashlib,json,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['original_domain_statistics_v1.py','original_domain_statistics_v2.py',
       'DOMAIN24_original_statistics_registration_v2.json','DOMAIN24_original_bootstrap_draws_v2.bin',
       'crosscheck_original_statistics_v1.py','DOMAIN24_original_statistics_independent_crosscheck_v1.json',
       'original_statistics_integrity_v1.py','DOMAIN24_original_statistics_integrity_v1.json',
       'audit_original_statistics_integrity_v1.py','DOMAIN24_original_statistics_integrity_synthetic_v1.json',
       'run_domain_original_raw_v3.py','upgrade_original_raw_runner_v3.py',
       'audit_original_raw_resume_v1.py','DOMAIN24_original_raw_resume_synthetic_audit_v1.json',
       'capture_original_runtime_v1.py','DOMAIN24_original_runtime_contract_v1.json',
       'register_original_raw_fit_v3.py','critique_DOMAIN24_original_statistics_v1.md']
assert all((HERE/n).exists() for n in names)
def read(n):return json.loads((HERE/n).read_text(encoding='utf-8'))
check=read('DOMAIN24_original_statistics_independent_crosscheck_v1.json')
assert check['draws_replayed']==200000 and len(check['synthetic_tests'])==8
integrity=read('DOMAIN24_original_statistics_integrity_v1.json')
assert integrity['draws_checked']==200000 and integrity['full_model_score_gate'] is False
assert len(read('DOMAIN24_original_statistics_integrity_synthetic_v1.json')['cases'])==10
assert len(read('DOMAIN24_original_raw_resume_synthetic_audit_v1.json')['tests'])==13
assert len(read('DOMAIN24_original_runtime_contract_v1.json')['model_contracts'])==9
folder=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2'
prep={p.name:sha(p) for p in folder.glob('*_fold*.json')}
record={'checkpoint':19,'files_sha256':{n:sha(HERE/n) for n in names},
        'preparation_receipts_snapshot':prep,'original_preparation_complete':(folder/'complete.json').exists(),
        'original_worker_handle':77114,'last_authoritative_poll':'running, P2LOO_fold26 PASS',
        'original_model_fit_registered':False,'original_model_fits':0,'original_heldout_score':False,
        'statistics_draws_replayed':200000,'statistics_boundary_tests':8,'statistics_file_negative_tests':10,
        'resume_contract_tests':13,'model_constructor_contracts':9,
        'statistics_full_model_gate':False,'independent_raw3_registrar_critique':'requested/pending at snapshot',
        'whole_goal_complete':False,'adoption':False,'GPU':False,'submission':False,
        'next':['Poll77114 until actualcomplete66','Execute stdlib full66 preparation crosscheck',
                'Review raw3/registrar3 actualplan before any fit registration',
                'Register all24x3 original raw cells then start CPU runner3',
                'Reference-only cached PFN66/fullpostprocessing/gates before originalheldoutscore',
                'All original validators, CH2 extension, literature24, data142/subset+interaction, firstunused confirmation/report remain']}
with (HERE/'checkpoint_record_v19.json').open('x',encoding='utf-8') as h:json.dump(record,h,ensure_ascii=False,indent=2)
note=f'''
# 현재 재개 지점 — 2026-10-07 집 코덱스 (checkpoint v19)

- 직전goalturn은도메인48 actualscore/Decimal/보고서·사후비평완료로progress. 이번도통계봉인·독립검사/합성resume/actualruntime계약실행으로progress.
- 통계v1 syntaxextra] exit1/파일보존→새v2actualexit0. 원dayfloor/5·87block(F13 42/F47 45) shared200k binary34800000byte 생성/등록2(정답값0). 원block/8640ID·TM2664/200k전체Counter재생+synthetic8 actualexit0. integrity전수 farm합·SHA/layout/Python3.12 actualPASS,손상/잘못된count등합성10PASS. 독립critic source5pin/draw일치·봉인core오류0.
- 원rawrunner3 source에actualySHA/modelcontract/실runtime/stat2/draw/crosscheck 연결. syntheticresume13actualPASS(학습금지/실query·정답0), 실제environment/modulepaths/9constructor계약capture actualexit0. registrar3 ASTPASS/미실행;66complete+독립prep전fit등록금지. 원raw3/registrarcritic요청중.
- 원77114 직접poll running/P2LOO_fold26 PASS,파일snapshot{len(prep)}/66/complete존재{record['original_preparation_complete']}. 같은handle부터재개·lock/파일로live추측하거나중복재시작금지.
- OS01/02 futurefullscorer는여전히필수: fullmodelgate/source/draw/layout검사후truthparse,24전체·두p교집합+3seed×TM/P2/EL방향 강제. 통계integrity는fullmodelgate아님. alpha84prospective탐색/oldBLK54·최종firstunused확정상위규칙유지.
- 다음원77114완료→독립full66prep→raw3source/등록최종비평→registrar3→CPU원24×3seed·594baseline/4752candidatefit. PFN66참조전용cache·전체mix/shrink/SG2/gates 별도필수. 문헌24/자료142·원열subset/interaction/CH2원검증/미사용확정1회·전체보고서미완료. goal active/GPU·채택·제출0.

이하 이전 시점 기록.

'''
progress=HERE/'PROGRESS.md';progress.write_text(note+progress.read_text(encoding='utf-8'),encoding='utf-8')
for p in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-07.md']:
    with p.open('a',encoding='utf-8') as h:h.write(note)
catalog=ROOT/'공용/데이터_단서_카탈로그.md'
assert max(int(x) for x in re.findall(r'\| 6\.(\d+) \|',catalog.read_text(encoding='utf-8')))==382
entry='''
| 6.383 | **원domain 통계20만표본 실행전봉인·독립전체재생 및 raw재개계약 실제감사** (2026-10-07 집 코덱스) | 원dayfloor/5 87block/shared200k binary34800000byte·DIAG8640/TM2664 정답값없이봉인;alpha.025/84 prospective탐색/기존BLK54소급변경0. independentCounter200k재생+synthetic8 actualPASS;statSHA/layout/Python3.12/farm합전수PASS·손상draw등합성10PASS. 원raw3 actualy/runtime/modelcontract연결·합성resume13 actualPASS/fit금지,actual9constructor/runtimecapture exit0. registrar3 ASTonly/미실행·66complete와독립검사전fit등록0;worker77114 directrunning/P2fold26,전체prep미완료. 통계integrity는fullmodelgate아님·future24전체/두p교집합/전seed방향scorer필수. 모든원validator/CH2원검증/문헌/자료/subset·interaction/최초미사용확정/전체보고서남음·goal active/GPU·채택·제출0 | 집/코덱스/analysis/ec_feature_rebuild_20261007_v1/DOMAIN24_original_statistics_registration_v2.json;DOMAIN24_original_statistics_independent_crosscheck_v1.json;DOMAIN24_original_statistics_integrity_v1.json;DOMAIN24_original_raw_resume_synthetic_audit_v1.json;DOMAIN24_original_runtime_contract_v1.json;checkpoint_record_v19.json |
'''
with catalog.open('a',encoding='utf-8') as h:h.write(entry)
print(f'checkpoint19/catalog6.383 actual progress persisted; originalprep {len(prep)}/66 incomplete')
