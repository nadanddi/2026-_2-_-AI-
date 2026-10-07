from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
folder=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2'
names=['register_original_raw_fit_v4.py','register_original_raw_fit_v5.py',
       'upgrade_original_registrar_v4.py','upgrade_original_registrar_v5.py',
       'critique_DOMAIN24_original_raw_registration_plan_v1.md',
       'critique_DOMAIN24_original_raw_registration_plan_v2.md']
record={'status':'PROSPECTIVE_FIT_REGISTRAR_AUDIT_LINKS_CLOSED_NOT_REGISTERED',
        'source_sha256':{n:sha(HERE/n) for n in names},'latest_registrar':'register_original_raw_fit_v5.py',
        'output_registration':'DOMAIN24_original_raw_fit_registration_v3.json','runner':'run_domain_original_raw_v3.py',
        'actual_O_negative_v4_exit_code':1,'actual_O_negative_v4_reason':'RuntimeError before registration reading',
        'registration_exists':(HERE/'DOMAIN24_original_raw_fit_registration_v3.json').exists(),
        'preparation_complete_exists':(folder/'complete.json').exists(),
        'preparation_receipts_snapshot':{p.name:sha(p) for p in folder.glob('*_fold*.json')},
        'worker77114_last_poll':'running/P2LOO_fold29 PASS','model_fits':0,'whole_goal_complete':False,
        'registrar5_source_critique':'requested/pending; v4 critique closed ORR01/ORR02',
        'runtime19_pins_independent_verification':'critic actual escalation PASS in critique plan v2'}
with (HERE/'DOMAIN24_original_registrar_followup_checkpoint_v19.json').open('x',encoding='utf-8') as h:
    json.dump(record,h,ensure_ascii=False,indent=2)
note=f'''
### v19 원학습등록 후속 — 최신 실행등록기는 v5
- ORR01 저장PASS와현감사sourceSHA연결누락/ORR02 -O허용을새v4로보완. actual-O negative exit1(RuntimeError파일읽기전)/등록생성0. critic planv2 actual확인 ORR01/02닫힘/runtime19핀정식승인독립검산전부MATCH.
- 추가inheritedextras핀충돌거부와nofit/no-target flags/runtime일치→새registrar5 source/AST PASS/독립diff요청중. 다음은 **register_original_raw_fit_v5.py** →등록파일v3 →run_domain_original_raw_v3.py. 과거v3/4보존,실등록/fit0.
- 원77114 마지막직접poll running/P2LOO_fold29 PASS. 현파일snapshot{len(record['preparation_receipts_snapshot'])}/66/complete존재{record['preparation_complete_exists']}. 완료뒤stdllib crosscheck_original_preparation_v1.py와최신등록기비평/등록필수. partialprep으로학습금지. goal active/전체원검증·PFN/후처리·문헌·자료·최종미사용1회/전체보고서미완료.
'''
progress=HERE/'PROGRESS.md';progress.write_text(note+'\n'+progress.read_text(encoding='utf-8'),encoding='utf-8')
for p in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-07.md']:
    with p.open('a',encoding='utf-8') as h:h.write(note)
print('v19 followup saved; latest registrar5, registration/fit0, goal incomplete')
