"""Version the unexecuted original-fold preparer with stricter receipt contracts."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'prepare_domain_original_all_v1.py').read_text(encoding='utf-8')
s=s.replace('DOMAIN24_original_preparation_registration_v1.json','DOMAIN24_original_preparation_registration_v2.json')
s=s.replace('DOMAIN24_ORIGINAL_PREPARATION_v1','DOMAIN24_ORIGINAL_PREPARATION_v2')
s=s.replace('    files={}','    files={}\n    prefix_checks_total=0\n    expected_files={f\'{f["validator"]}_fold{f["fold"]}.json\' for f in registry["folds"]}\n    assert len(expected_files)==len(registry["folds"])==66')
old="                assert saved['status']=='PREPARED_FEATURES_NOT_FIT_OR_SCORE'"
new=old+"""
                assert saved['validator']==fold['validator'] and saved['fold']==fold['fold']
                assert saved['domain_columns']==1187
                assert set(saved)=={'status','registration_sha256','validator','fold','ordered_train_ids',
                    'ordered_query_ids','input_forbidden_ids','candidate_matrices','prefix_checks','domain_columns',
                    'heldout_truth_loaded','model_fit','performance_evaluated','duration_seconds'}
"""
assert old in s;s=s.replace(old,new)
old="                    assert signature['columns']==FULL_R3+reg['family_map'][cid]['additional_ET_columns']"
new=old+"""
                    assert set(signature)=={'columns','train_sha256','query_sha256','all_missing_train_columns'}
                    empty=signature['all_missing_train_columns']
                    assert isinstance(empty,list) and len(empty)==len(set(empty)) and set(empty)<=set(signature['columns'])
"""
assert old in s;s=s.replace(old,new)
s=s.replace('                files[name]=sha(path)','                files[name]=sha(path)\n                prefix_checks_total+=saved["prefix_checks"]')
s=s.replace('            files[name]=sha(path)','            files[name]=sha(path)\n            prefix_checks_total+=checks')
# The latter substring also matches the deeper-indented resume line. Repair only that exact overlap.
s=s.replace('                files[name]=sha(path)\n            prefix_checks_total+=checks\n                prefix_checks_total+=saved["prefix_checks"]',
            '                files[name]=sha(path)\n                prefix_checks_total+=saved["prefix_checks"]')
s=s.replace('        assert len(files)==66','        assert set(files)==expected_files and prefix_checks_total==396\n        assert {p.name for p in folder.glob("*_fold*.json")}==expected_files')
s=s.replace("'folds':66,'prefix_checks':396","'folds':66,'prefix_checks':prefix_checks_total")
compile(s,'prepare_domain_original_all_v2.py','exec')
out=HERE/'prepare_domain_original_all_v2.py';assert not out.exists()
out.write_text(s,encoding='utf-8')
print('Original66 preparation v2 source created; not registered/executed')
