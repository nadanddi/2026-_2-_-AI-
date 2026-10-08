"""Preregister conditional Decimal verifier with full assembly source chain."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    priorpath = HERE/'ORIGINAL_DOMAIN24_pipeline_registration_v3.json'
    prior = json.loads(priorpath.read_text(encoding='utf-8'))
    assert prior['status'] == 'REGISTERED_ORIGINAL66_DOMAIN24_ASSEMBLY_NO_SCORE'
    pins = dict(prior['sources_sha256'])
    assert all(sha(p) == value for p,value in pins.items())
    pins[str(priorpath.resolve())] = sha(priorpath)
    additions = ('verify_original_domain_arithmetic_v3.py','original_independent_arithmetic_v1.py',
                 'audit_original_independent_arithmetic_v2.py','original_independent_arithmetic_synthetic_audit_v2.json',
                 'original_choice_snapshot_validation_v1.py','audit_original_choice_snapshot_v1.py',
                 'original_choice_snapshot_synthetic_audit_v1.json','register_original_domain_arithmetic_v3.py',
                 'critique_ORIGINAL_independent_arithmetic_v1.md',
                 'critique_ORIGINAL_arithmetic_driver_plan_v1.md',
                 'critique_ORIGINAL_arithmetic_driver_plan_v3.md',
                 'critique_ORIGINAL_domain_assembly_registration_v3.md')
    for name in additions:
        path = (HERE/name).resolve(); value = sha(path)
        if str(path) in pins: assert pins[str(path)] == value
        pins[str(path)] = value
    arithmetic = json.loads((HERE/'original_independent_arithmetic_synthetic_audit_v2.json').read_text(encoding='utf-8'))
    assert arithmetic['status'] == 'INDEPENDENT_RECIPE_SYNTHETIC_PASS_NOT_FULL_GATE'
    assert arithmetic['decimal_precision'] == 60 and arithmetic['decimal_stage_comparisons'] == 384
    assert arithmetic['invalid_count'] == len(arithmetic['invalid_checks']) == 14
    assert arithmetic['prediction_prefix_poison_cases'] == 4
    assert arithmetic['whole_pipeline_gate_passed'] is arithmetic['heldout_truth_read'] is arithmetic['model_fit'] is False
    sources3 = {str((HERE/n).resolve()):sha(HERE/n) for n in
                ('audit_original_independent_arithmetic_v2.py','original_independent_arithmetic_v1.py',
                 'original_postprocess_kernel_v2.py')}
    assert arithmetic['executed_sources_sha256'] == sources3
    metadata = json.loads((HERE/'original_choice_snapshot_synthetic_audit_v1.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 'SYNTHETIC_CHOICE_METADATA_PASS_NOT_SELECTION_OR_FULL_GATE'
    assert metadata['positive_checks'] == 1 and metadata['invalid_count'] == len(metadata['invalid_checks']) == 19
    assert metadata['whole_pipeline_gate_passed'] is metadata['heldout_truth_read'] is metadata['model_fit'] is False
    sources2 = {str((HERE/n).resolve()):sha(HERE/n) for n in
                ('audit_original_choice_snapshot_v1.py','original_choice_snapshot_validation_v1.py')}
    assert metadata['executed_sources_sha256'] == sources2
    assert all(pins[path] == value for path,value in {**sources3,**sources2}.items())
    choice_paths = [(HERE/n).resolve() for n in ('original_sg2_plan_v1.py','blk_sg2_refonly_v1.py',
                                              'blk_sg2_refonly_v2.py','original_postprocess_kernel_v2.py')]
    choice_paths += [(HERE.parents[3]/'집/클로드/submission14_ec_sg2/sg2post.py').resolve()]
    choices5 = {str(path):sha(path) for path in choice_paths}
    assert len(choices5) == 5 and all(pins[path] == value for path,value in choices5.items())
    result = {'status':'REGISTERED_ORIGINAL66_CONDITIONAL_ARITHMETIC_NO_SCORE',
              'sources_sha256':pins,'SG2_choice_sources_sha256':choices5,
              'verifier_sha256':sha(HERE/'verify_original_domain_arithmetic_v3.py'),
              'required_manifests':['raw5346 original66','PFN528 original66','assembled132 original66'],
              'decimal_precision':60,'absolute_tolerance':1e-12,
              'SG2_threshold_ambiguity_policy':'Fail closed when abs(abs(delta)-0.30)<=1e-12; investigate via independent source replay, no tolerance adjustment',
              'scope':'All66 folds × 3 seeds × baseline+24 candidates × 4 stored stages',
              'resume':'Fresh whole fold arithmetic recomputation, strict receipt equality, O_EXCL singlewriter',
              'final_hashes':'producer outputs/root/fold completion + assembly outputs/complete + arithmetic66 + source pins',
              'model_fit':False,'heldout_truth_read':False,'whole_pipeline_gate_passed':False,
              'score_permitted':False,
              'limits':['Conditional on saved SG2 choices/train bounds; no independently rebuilt selection here',
                        'Independent full feature/label lineage and actual query-input causality verifier still mandatory']}
    with (HERE/'ORIGINAL_DOMAIN24_arithmetic_registration_v3.json').open('x',encoding='utf-8') as out:
        json.dump(result,out,ensure_ascii=False,indent=2)
    print('Original66 conditional arithmetic registered',len(pins),'pins; no actual66 replay or score')


if __name__ == '__main__': main()
