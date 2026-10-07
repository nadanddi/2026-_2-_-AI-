"""Close stale audit PASS linkage in a new prospective fit registrar."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
source=(HERE/'register_original_raw_fit_v3.py').read_text(encoding='utf-8')
source=source.replace('import hashlib,json,ast','import hashlib,json,ast,sys')
source=source.replace('def main():\n',"def main():\n    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited before fit registration')\n",1)
source=source.replace("    assert prepcheck['status']=='INDEPENDENT_ORIGINAL66_PREPARATION_METADATA_PASS'",
    "    assert prepcheck['code_sha256']==sha(HERE/'crosscheck_original_preparation_v1.py')\n    assert prepcheck['status']=='INDEPENDENT_ORIGINAL66_PREPARATION_METADATA_PASS'")
source=source.replace("    assert runtime['status']=='ACTUAL_ORIGINAL_MODEL_RUNTIME_CONSTRUCTORS_CAPTURED_NO_FIT'",
    "    assert runtime['code_sha256']==sha(HERE/'capture_original_runtime_v1.py')\n    assert runtime['status']=='ACTUAL_ORIGINAL_MODEL_RUNTIME_CONSTRUCTORS_CAPTURED_NO_FIT'")
source=source.replace("    assert synthetic['runner_sha256']==runtime['runner_sha256']",
    "    assert synthetic['code_sha256']==sha(HERE/'audit_original_raw_resume_v1.py')\n    assert synthetic['runner_sha256']==runtime['runner_sha256']")
old="    assert statscheck['status']=='INDEPENDENT_ORIGINAL_STATISTICS_DRAW_AND_SYNTHETIC_PASS'"
new="""    assert statscheck['code_sha256']==sha(HERE/'crosscheck_original_statistics_v1.py')
    assert statscheck['blocks_checked']==87 and statscheck['row_membership_checked']==8640
    assert all(t['PASS'] for t in statscheck['synthetic_tests'])
    integrity=read(HERE/'DOMAIN24_original_statistics_integrity_v1.json')
    negative=read(HERE/'DOMAIN24_original_statistics_integrity_synthetic_v1.json')
    assert integrity['code_sha256']==sha(HERE/'original_statistics_integrity_v1.py')
    assert integrity['statistics_registration_sha256']==sha(HERE/'DOMAIN24_original_statistics_registration_v2.json')
    assert integrity['draw_sha256']==stats['draw_file_sha256'] and integrity['draws_checked']==200000
    assert integrity['full_model_score_gate'] is False and integrity['heldout_truth_loaded'] is False
    assert negative['code_sha256']==sha(HERE/'audit_original_statistics_integrity_v1.py')
    assert negative['guard_sha256']==integrity['code_sha256']
    assert len(negative['cases'])==10 and all(t['PASS'] for t in negative['cases'])
    assert statscheck['status']=='INDEPENDENT_ORIGINAL_STATISTICS_DRAW_AND_SYNTHETIC_PASS'"""
assert source.count(old)==1;source=source.replace(old,new)
old="        'register_original_raw_fit_v3.py']"
new="""        'register_original_raw_fit_v3.py','register_original_raw_fit_v4.py','upgrade_original_registrar_v4.py',
        'original_statistics_integrity_v1.py','DOMAIN24_original_statistics_integrity_v1.json',
        'audit_original_statistics_integrity_v1.py','DOMAIN24_original_statistics_integrity_synthetic_v1.json']"""
assert source.count(old)==1;source=source.replace(old,new)
ast.parse(source)
with (HERE/'register_original_raw_fit_v4.py').open('x',encoding='utf-8') as h:h.write(source)
print('Registrar4 source closes recorded-auditor/currentSHA links, integrity links and -O; no registration/fit')
