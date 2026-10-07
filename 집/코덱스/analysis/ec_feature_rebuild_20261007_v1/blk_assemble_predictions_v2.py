"""Same frozen mix plus strict cross-member training/context/weight lineage."""
from pathlib import Path
import json,hashlib,math
HERE=Path(__file__).resolve().parent
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def preflight():
    r=HERE/'checkpoints/BLK_R3_v1';p=HERE/'checkpoints/BLK_PFN_CPU_v1'
    assert (r/'complete.json').exists() and (p/'complete.json').exists(),'preserve original live CPU worker'
    rr=json.loads((r/'registration.json').read_text(encoding='utf-8'))
    pr=json.loads((p/'registration.json').read_text(encoding='utf-8'))
    assert pr['R3_registration_sha256']==_sha(r/'registration.json')
    assert pr['query_ids']==rr['ordered_query_ids']
    assert pr['columns']==rr['feature_columns']['PFN_NOT_EXECUTED']
    assert pr['contexts']==rr['PFN_context_ids_NOT_EXECUTED']
    assert _sha(Path(pr['weights_path']))==pr['weights_sha256']=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    assert pr['code_sha256']==_sha(HERE/'blk_pfn_baseline_v1.py')
    assert pr['environment']['device']=='cpu' and pr['n_estimators']==4 and pr['precision']=='float32'
    assert pr['model_version']=='V2' and not pr['heldout_truth_loaded']
    ids=rr['ordered_query_ids'];train=set(rr['ordered_train_ids'])
    assert len(ids)==1440 and len(train)==5520 and not(train&set(ids))
    assert set(pr['contexts'])=={'5','6','7','8'}
    assert all(len(v)==2000 and len(set(v))==2000 and set(v)<=train for v in pr['contexts'].values())
    paths=[r/f'{n}_seed{s}.json' for n in ['ET','LGB','MLP'] for s in [47,1414,6464]]+[p/f'context{s}.json' for s in [5,6,7,8]]
    for file in paths:
        data=json.loads(file.read_text(encoding='utf-8'))
        assert data['row_ids']==ids and len(data['pred'])==len(ids)
        assert all(math.isfinite(x) for x in data['pred'])
        assert data['registration_sha256']==_sha(file.parent/'registration.json')
    return {'R3_registration_sha256':_sha(r/'registration.json'),'PFN_registration_sha256':_sha(p/'registration.json'),
        'member_files_sha256':{str(file.relative_to(HERE)):_sha(file) for file in paths},
        'weights_sha256':pr['weights_sha256'],'assembler_sources_sha256':{n:_sha(HERE/n) for n in ['blk_assemble_predictions_v1.py','blk_assemble_predictions_v2.py']},
        'cross_training_context_contract_verified':True}

if __name__=='__main__':
    lineage=preflight()
    code=(HERE/'blk_assemble_predictions_v1.py').read_text(encoding='utf-8')
    code=code.replace('checkpoints/BLK_ASSEMBLED_v1','checkpoints/BLK_ASSEMBLED_v2')
    code=code.replace("manifest={'status':", "manifest={'lineage':lineage,'status':")
    exec(compile(code,str(Path(__file__).resolve()),'exec'),globals())
