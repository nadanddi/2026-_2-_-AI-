"""Fresh file replay and transitive source checks for the CH2 training reference."""
from pathlib import Path
import json,hashlib,sys
import ch2_prefix_sources_v3 as prefix_module

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def canonical_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def load_reference():
    if not __debug__ or sys.flags.optimize:
        raise RuntimeError('Optimized Python disables assertion guards and is prohibited')
    assert Path(prefix_module.__file__).resolve()==(HERE/'ch2_prefix_sources_v3.py').resolve()
    prep=read('CH2_training_reference_preparation_v1.json')
    assert prep['status']=='TRAIN_REFERENCE_PREPARATION_PASS_NOT_QUERY_OR_PERFORMANCE_GATE'
    assert prep['query_rows_numerically_parsed']==prep['gap_rows_numerically_parsed']==prep['heldout_labels_numerically_parsed']==0
    assert not prep['performance_evaluated'] and not prep['model_fit']
    audit=read('CH2_reference_independent_audit_v1.json')
    synthetic=read('CH2_prefix_synthetic_audit_v2.json')
    pins={}
    for receipt in [prep,audit,synthetic]:
        sources=receipt.get('source_sha256',receipt.get('sources_sha256'))
        assert sources
        for path,value in sources.items():
            assert path not in pins or pins[path]==value
            assert sha(path)==value
            pins[path]=value
    stored=HERE/'CH2_training_reference_v1.json'
    assert sha(stored)==prep['reference_file_sha256']
    payload=json.loads(stored.read_text(encoding='utf-8'))
    assert canonical_digest(payload)==prep['reference_payload_digest']
    assert payload['source_sha256']==prep['source_sha256']
    layout=read('BLK_layout_v2.json')
    assert payload['layout_sha256']==sha(HERE/'BLK_layout_v2.json')
    graph=read('CH2_reference_diagnostics_v1.json')
    assert payload['links']==graph['links']
    assert len(payload['records'])==audit['train_records']==prep['train_records']
    assert len(payload['links'])==audit['links_checked']==prep['links']
    records={}
    for node in payload['records']:
        key=node['farm'],node['day'];assert key not in records
        records[key]={int(h):obs for h,obs in node['hours'].items()}
    scales={v['farm']:v['weather_scales_train_only'] for v in audit['farms']}
    assert payload['weather_scales']==scales
    model=prefix_module.PrefixSourceCH2(layout,records,payload['links'],payload['weather_scales'])
    assert {f:dict(v) for f,v in model.scales.items()}==scales
    assert len(model.cyclic_roots)==prep['cyclic_components']
    # Exact raw-reference equality survives actual file read, not just an in-memory encoding.
    assert all(dict(model.records[node][h])==obs for node,hours in records.items() for h,obs in hours.items())
    assert all(sha(path)==value for path,value in pins.items())
    return model,layout,{'direct_and_transitive_source_sha256':pins,
                        'reference_file_sha256':sha(stored),'reference_payload_digest':canonical_digest(payload),
                        'import_path':str(Path(prefix_module.__file__).resolve()),
                        'import_source_sha256':sha(prefix_module.__file__),
                        'Python_optimization':sys.flags.optimize,'scales_explicit_replay_exact':True}

if __name__=='__main__':
    model,layout,details=load_reference()
    result={'status':'FRESH_TRAIN_REFERENCE_FILE_REPLAY_PASS_NO_QUERY_OR_SCORE',**details,
            'loader_source_sha256':sha(__file__),'train_rows':len(layout['train_ids']),
            'train_records':len(model.records),'cyclic_components':len(model.cyclic_roots),
            'query_inputs_parsed':0,'heldout_labels_parsed':0,'performance_evaluated':False}
    output=HERE/'CH2_training_reference_file_replay_v1.json';assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Actual stored-file replay PASS; transitive sources/import path/scales verified; no query/score')
