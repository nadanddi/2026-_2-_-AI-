"""Pure-standard-library registration of feature preparation, not model experiments."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,ast
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    pipeline=read(HERE/'DOMAIN24_BLK_pipeline_registration_v2.json')
    sources=dict(pipeline['sources_sha256'])
    assert all(sha(path)==value for path,value in sources.items())
    fit=read(HERE/'DOMAIN24_BLK_fit_registration_v3.json')
    registry=read(HERE/'original_validator_endpoint_registry_v1.json')
    original=read(HERE/'fold_registry_v1.json')
    integrity=read(HERE/'original_validator_integrity_audit_v2.json')
    assert integrity['status']=='PASS'
    assert integrity['registry_sha256']==sha(HERE/'original_validator_endpoint_registry_v1.json')
    assert len(registry['folds'])==len(original['folds'])==66
    counts=Counter();rows=Counter();records=[]
    for fold,base in zip(registry['folds'],original['folds']):
        assert (fold['validator'],fold['fold'])==(base['validator'],base['fold'])
        assert set(fold['ordered_query_ids'])==set(base['query_ids'])
        train=fold['ordered_train_ids'];query=fold['ordered_query_ids'];forbidden=fold['input_forbidden_ids']
        assert train==sorted(train) and query==sorted(query)
        assert len(train)==len(set(train)) and len(query)==len(set(query))
        assert not(set(train)&set(query) or set(train)&set(forbidden) or set(query)&set(forbidden))
        assert {rid[:7] for rid in train}.isdisjoint(rid[:7] for rid in query)
        assert len(set(train)|set(query)|set(forbidden))==9600
        hours=defaultdict(set)
        for rid in train+query:hours[rid[:7]].add(int(rid[-2:]))
        assert all(h==set(range(24)) for h in hours.values())
        counts[fold['validator']]+=1;rows[fold['validator']]+=len(query)
        records.append({'validator':fold['validator'],'fold':fold['fold'],
                        'train_rows':len(train),'query_rows':len(query),'input_forbidden_rows':len(forbidden)})
    assert dict(counts)=={'DIAG10':10,'P2LOO':46,'EL1':10}
    assert dict(rows)=={'DIAG10':8640,'P2LOO':1104,'EL1':1104}
    tm={tuple(v) for v in original['TM_days']}
    tm_ids={rid for f in registry['folds'] if f['validator']=='DIAG10' for rid in f['ordered_query_ids']
            if (rid.split('_')[0],int(rid.split('_')[1])) in tm}
    assert len(tm)==111 and len(tm_ids)==2664
    extras=['original_validator_endpoint_registry_v1.json','fold_registry_v1.json',
        'original_validator_integrity_audit_v2.json','DOMAIN24_original_feature_probe_v1.json',
        'prepare_domain_original_probe_v1.py','prepare_domain_original_all_v2.py',
        'prepare_domain_original_all_upgrade_v2.py','register_domain_original_preparation_v2.py',
        'critique_DOMAIN24_original_preparation_v1.md']
    for name in extras:
        path=HERE/name;sources[str(path.resolve())]=sha(path)
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
    assert all(sha(path)==value for path,value in sources.items())
    result={'status':'REGISTERED_ORIGINAL_FEATURE_PREPARATION_NOT_MODEL_FIT',
        'runner_sha256':sha(HERE/'prepare_domain_original_all_v2.py'),
        'registry_sha256':sha(HERE/'original_validator_endpoint_registry_v1.json'),
        'sources_sha256':sources,'environment':read(HERE/'checkpoints/BLK_R3_v1/registration.json')['environment'],
        'folds':records,'fold_counts':dict(counts),'query_rows_per_validator':dict(rows),
        'TM_days':111,'TM_query_rows':2664,'family_map':fit['family_map'],
        'candidate_cap':24,'prefix_checks_per_fold':6,'required_prefix_checks':396,
        'matrix_dtype':'little-endian float64; canonical NaN bytes; row order from registry',
        'model_fit_registered':False,'performance_tested':False,'adoption_permitted':False,
        'resume_policy':'Metadata/source/payload verification only; every model fit must freshly reconstruct matrix hashes',
        'limits':['Feature preparation only, not original-validator model validation',
                  'Current CPU cached baseline needs separate model-fit registration and full per-fold audits',
                  'All24 domain candidates remain mandatory regardless of BLK ranking; both heldout labels hidden']}
    out=HERE/'DOMAIN24_original_preparation_registration_v2.json';assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Original66 feature-only preparation registered: {len(sources)} source hashes; no model fit/score',flush=True)

if __name__=='__main__':main()
