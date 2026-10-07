"""Freeze postprocess/scoring sources without altering any fit-time rule."""
from pathlib import Path
import json,hashlib,ast
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
fitpath=HERE/'DOMAIN24_BLK_fit_registration_v3.json'
fit=json.loads(fitpath.read_text(encoding='utf-8'))
assert fit['status']=='REGISTERED_BEFORE_DOMAIN_FIT'
assert all(sha(path)==value for path,value in fit['source_sha256'].items())
stages={'raw_receipt':'domain_raw_receipt_v1.py','assemble':'assemble_domain_BLK_v2.py',
        'verify':'verify_domain_BLK_v1.py','score':'score_domain_BLK_v1.py'}
sources=dict(fit['source_sha256'])
files=list(stages.values())+['domain_sg2_plan_v1.py','execute_domain_BLK_stage_v1.py',
                           'register_domain_BLK_pipeline_v2.py','DOMAIN24_BLK_fit_registration_v3.json',
                           'critique_DOMAIN24_postprocess_plan_v1.md']
for name in files:
    path=HERE/name
    if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'))
    sources[str(path.resolve())]=sha(path)
result={'status':'REGISTERED_BEFORE_DOMAIN_SCORE','fit_registration_sha256':sha(fitpath),
        'statistics':fit['statistics'],'candidate_ids':sorted(fit['family_map']),
        'stages':stages,'sources_sha256':sources,'executor_sha256':sha(HERE/'execute_domain_BLK_stage_v1.py'),
        'postprocess':'raw .6R3+.4PFN -> one prefix shrink -> train clip -> frozen ref-only SG2 -> train clip',
        'SG2_plan':'input-only inference selection frozen; prediction mean/guard/correction recomputed unchanged',
        'required_audits':{'baseline_SG2_all_rows':8640,'candidate_source_and_future_checks':13824,
                           'independent_whole_scalar_checks':518400,'raw_files':75},
        'adoption_permitted':False,'heldout_truth_loaded_for_domain_score':False,
        'limits':['Pipeline implemented after raw fit start; all fit-time rules/statistics remain unchanged',
                  'Full72 completion and registered raw receipt/assemble/verify gates required before scoring',
                  'Source and finite empirical audits; no universal input-proof or independently refit ET claim',
                  'All24 original TM/P2LOO/EL1 checks remain; BLK is exposed exploratory diagnosis']}
out=HERE/'DOMAIN24_BLK_pipeline_registration_v2.json';assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(f'Domain pipeline sources registered: {len(sources)} files, frozen fit-time statistics unchanged',flush=True)
