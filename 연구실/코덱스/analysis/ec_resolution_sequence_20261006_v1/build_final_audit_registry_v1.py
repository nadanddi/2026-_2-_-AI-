from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent
patterns={
 'stage1_complete':('verify_stage1_v1.py','verify_stage1_complete_*.json'),
 'summary_and_explanation':('verify_stage1_summary_explanation_v1.py','verify_stage1_summary_explanation_v1_*.json'),
 'nested_risk_inputs':('verify_risk_inputs_v1.py','verify_risk_inputs_v1_*.json'),
 'legacy_helper_defect':('verify_risk_core_contract_v1.py','verify_risk_core_contract_v1_*.json'),
 'corrected_helper':('verify_risk_core_contract_v2.py','verify_risk_core_contract_v2_*.json'),
 'real_feature_boundary':('verify_risk_real_design_v1.py','verify_risk_real_design_v1_*.json'),
 'risk_fit_correction_screen':('verify_risk_result_v1.py','verify_risk_result_v1_*.json'),
 'posthoc_concentration':('verify_risk_diagnosis_v2.py','verify_risk_diagnosis_v2_*.json'),
 'coverage_and_case_exclusion':('verify_risk_scope_v3.py','verify_risk_scope_v3_*.json')}
checks=[]
for name,(code,pattern) in patterns.items():
    p=sorted(H.glob(pattern))[-1];result=json.loads(p.read_text(encoding='utf-8'));assert result['status'].startswith('PASS')
    checks.append(dict(scope=name,code=str(H/code),code_sha256=hashlib.sha256((H/code).read_bytes()).hexdigest(),receipt=str(p),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status=result['status'],expected_legacy_defect=name=='legacy_helper_defect'))
out=dict(status='FINAL_AUDIT_COMPLETE_CANDIDATE_REJECTED',time=datetime.datetime.now().isoformat(),checks=checks,audit_new_fit=0,candidate_status='SCREEN_REJECT',reject_reason='101:high:protection_failed',pass2='UNTESTED_ZERO_ROWS',baseline='original EC season-v2, not current EC14',full80=False,adoption=False,new_candidate=False,parameters_retuned=False)
with (H/'최종감사_목록_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('FINAL AUDIT REGISTRY',len(checks),out['status'])
