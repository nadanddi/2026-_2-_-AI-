"""First-audit digest guard; preserve v2 and all prepared artifacts."""
from pathlib import Path
H=Path(__file__).resolve().parent
target=H/'run_v3.py'; assert not target.exists()
s=(H/'run_v2.py').read_text(encoding='utf-8')
s=s.replace("preparation_v2.json","preparation_v3.json")
needle="if len(files)==3:validate_first(json.loads(first.read_text(encoding='utf-8')),signature)"
assert s.count(needle)==1
s=s.replace(needle,"if len(files)==3:\n                    assert saved['first_audit_sha256']==S.sha(first)\n                    validate_first(json.loads(first.read_text(encoding='utf-8')),signature)")
needle="""                S.savej(meta,saved)
                if check is not None:
                    first_record=dict(**check,signature=signature);validate_first(first_record,signature);S.savej(first,first_record)
"""
assert s.count(needle)==1
s=s.replace(needle,"""                if check is not None:
                    first_record=dict(**check,signature=signature);validate_first(first_record,signature);S.savej(first,first_record)
                    saved['first_audit_sha256']=S.sha(first)
                S.savej(meta,saved)
""")
target.write_text(s,encoding='utf-8')
print('V3_CREATED_NO_FIT')
