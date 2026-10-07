"""Preserve unexecuted v1; strengthen resume validation in a new prospective runner."""
from pathlib import Path
import ast

HERE = Path(__file__).resolve().parent
source = (HERE / 'run_domain_original_raw_v1.py').read_text(encoding='utf-8')
old = """    path=folder/f'{candidate}_seed{seed}.json'
    if path.exists():"""
new = """    expected_imputer={}
    if name in ['ET','MLP']:
        with np.errstate(invalid='ignore'):
            direct=np.nanmedian(tx.to_numpy(float),axis=0)
        empty=[c for c in tx if tx[c].isna().all()]
        expected_imputer={'statistics_sha256':canonical_numeric_sha(direct),
            'independent_train_median_sha256':canonical_numeric_sha(direct),
            'all_missing_train_columns':empty,
            'effective_columns':[c for c in tx if c not in empty],'fit_reference_only':True}
    path=folder/f'{candidate}_seed{seed}.json'
    if path.exists():"""
assert source.count(old) == 1
source = source.replace(old, new)
old = """        assert saved['audit']['all_errors_max']<=1e-6
        return path"""
new = """        assert saved['imputer']==expected_imputer
        audit=saved['audit']
        errors=audit['reverse_scattered_prefix_max_differences']
        assert len(errors)==2+len(probes) and all(np.isfinite(errors))
        assert all(e>=0 for e in errors) and audit['all_errors_max']==max(errors)<=1e-6
        assert audit['fit_rows']==len(tx) and audit['query_rows']==len(qx)
        verify_sources(reg)
        return path"""
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace("hashlib.sha256(transformer.statistics_.tobytes()).hexdigest()", "canonical_numeric_sha(transformer.statistics_)")
source = source.replace("hashlib.sha256(direct.tobytes()).hexdigest()", "canonical_numeric_sha(direct)")
old = """    verify_sources(reg)
    record={'status'"""
new = """    assert imputer==expected_imputer
    verify_sources(reg)
    record={'status'"""
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace("def verify_sources(reg):", """def canonical_numeric_sha(values):
    values=np.asarray(values,dtype='<f8').copy()
    values[np.isnan(values)]=np.nan
    return hashlib.sha256(values.tobytes()).hexdigest()

def verify_sources(reg):""")
source = source.replace('DOMAIN24_original_raw_fit_registration_v1.json', 'DOMAIN24_original_raw_fit_registration_v2.json')
source = source.replace('DOMAIN24_ORIGINAL_RAW_v1', 'DOMAIN24_ORIGINAL_RAW_v2')
old = """        complete=root/'complete.json';assert not complete.exists()
        atomic(complete,json.dumps({'status':'ORIGINAL66_R3_DOMAIN_RAW_COMPLETE_NO_FULL_BASELINE_OR_SCORE',"""
new = """        complete=root/'complete.json'
        summary={'status':'ORIGINAL66_R3_DOMAIN_RAW_COMPLETE_NO_FULL_BASELINE_OR_SCORE',"""
assert source.count(old) == 1
source = source.replace(old,new)
old = """                                                'original TM/P2LOO/EL1 scoring and independent verification']},ensure_ascii=False))"""
new = """                                                'original TM/P2LOO/EL1 scoring and independent verification']}
        if complete.exists():assert json.loads(complete.read_text(encoding='utf-8'))==summary
        else:atomic(complete,json.dumps(summary,ensure_ascii=False))"""
assert source.count(old) == 1
source = source.replace(old,new)
ast.parse(source)
out=HERE/'run_domain_original_raw_v2.py'
with out.open('x',encoding='utf-8') as handle:handle.write(source)
print('Prospective runner2 created, resume imputer/audit/full-complete strengthened; no model fit registration or execution')
