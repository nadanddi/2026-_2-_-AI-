"""Actual-source pin checks on every boundary audit before the whole-model gate."""
from pathlib import Path
import json
from verify_BLK_baseline_v1 import read,sha,HERE,ROOT

def preflight_boundaries():
    sg=read(HERE/'BLK_SG2_refonly_audit_v2.json')
    assert sg['status']=='PASS' and sg['query_rows']==1440 and sg['reference_fit_rows']==5520
    assert sg['source_sha256']==sha(ROOT/'집/클로드/submission14_ec_sg2/sg2post.py')
    assert isinstance(sg['adapter_sha256'],dict)
    assert all(sha(HERE/n)==s for n,s in sg['adapter_sha256'].items())
    edge=read(HERE/'BLK_SG2_edge_audit_v1.json')
    assert edge['status']=='PASS' and all(sha(HERE/n)==s for n,s in edge['code_sha256'].items())
    boundary=read(HERE/'BLK_endpoint_boundary_audit_v2.json')
    old_rules=HERE/'BLK_method_registration_v2.json'
    assert boundary['registration_sha256']==sha(old_rules)
    rules=read(old_rules)
    for n in ['blk_endpoint_methods_v1.py','blk_endpoint_methods_v2.py']:
        assert sha(HERE/n)==rules['method_code_sha256'][n]
    return {n:sha(HERE/n) for n in ['verify_BLK_baseline_v1.py','verify_BLK_baseline_v2.py','BLK_method_registration_v2.json',
        'blk_sg2_audit_v2.py','blk_sg2_edge_audit_v1.py','register_endpoint_methods_v2.py']}

if __name__=='__main__':
    verifier_lineage=preflight_boundaries()
    source=(HERE/'verify_BLK_baseline_v1.py').read_text(encoding='utf-8')
    source=source.replace("receipt={'status':", "receipt={'verifier_lineage_sha256':verifier_lineage,'status':")
    source=source.replace('BLK_verified_baseline_receipt_v1.json','BLK_verified_baseline_receipt_v2.json')
    exec(compile(source,str(Path(__file__).resolve()),'exec'),globals())
