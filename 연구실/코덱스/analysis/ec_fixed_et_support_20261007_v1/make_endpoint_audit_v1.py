from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'audit_v1.py').read_text(encoding='utf-8')
a=s.index('    records=[]');b=s.index('    endpoint_checks=[]')
s=s[:a]+'''    tree=json.loads((H/'audit_tree_v1.json').read_text(encoding='utf-8'));assert tree['status']=='PASS';records=tree['new_seed7_fullforests']
    for r in records:assert r['forest_sha']==R.B.sha(R.L/(r['model']+'.npz'))
'''+s[b:]
s=s.replace("H/'audit_v1.json'","H/'audit_endpoint_v2.json'")
p=H/'audit_endpoint_v2.py';assert not p.exists();p.write_text(s,encoding='utf-8')
