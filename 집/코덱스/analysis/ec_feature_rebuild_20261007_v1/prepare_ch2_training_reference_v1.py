"""Freeze the audited original CH2 training graph; no query numerical parsing or score."""
from pathlib import Path
import csv,json,hashlib,math
from ch2_prefix_sources_v3 import PrefixSourceCH2,RAW,WEATHER

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'공용/대회자료/정형데이터/참가자_배포'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def main():
    layout=load('BLK_layout_v2.json')
    graph=load('CH2_reference_diagnostics_v1.json')
    audit=load('CH2_reference_independent_audit_v1.json')
    synthetic=load('CH2_prefix_synthetic_audit_v2.json')
    assert audit['status']=='TRAIN_ONLY_CH2_COST_ASSIGNMENT_AUDIT_PASS_NOT_PREDICTIVE_VALIDATION'
    assert audit['query_inputs_parsed']==audit['heldout_labels_parsed']==audit['gap_inputs_parsed']==0
    assert synthetic['check_count']==48 and not synthetic['performance_evaluated']
    for receipts in [audit,synthetic]:
        pins=receipts.get('source_sha256',receipts.get('sources_sha256'))
        assert pins and all(sha(p)==h for p,h in pins.items())
    assert graph['layout_sha256']==sha(HERE/'BLK_layout_v2.json')
    assert all(sha(DATA/name)==value for name,value in layout['source_sha256'].items())
    paths=[Path(__file__),HERE/'ch2_prefix_sources_v3.py',HERE/'BLK_layout_v2.json',
           HERE/'CH2_reference_diagnostics_v1.json',HERE/'CH2_reference_independent_audit_v1.json',
           HERE/'CH2_prefix_synthetic_audit_v2.json',DATA/'train_X.csv',DATA/'train_y.csv']
    pins={str(p.resolve()):sha(p) for p in paths}
    train=set(layout['train_ids']);query=set(layout['query_ids']);gap=set(layout['gap_ids_REMOVE_INPUT_AND_BOTH_LABELS'])
    assert not(train&query or train&gap or query&gap)
    inputs={};labels={}
    # CSV/hash read all strings/bytes. Convert numbers only after exact train-ID filtering.
    with (DATA/'train_X.csv').open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            rid=row['row_id']
            if rid not in train:continue
            assert rid not in inputs
            inputs[rid]={c:float(row[c]) if row[c].strip() else None for c in RAW}
    with (DATA/'train_y.csv').open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            rid=row['row_id']
            if rid not in train:continue
            assert rid not in labels
            labels[rid]=float(row['sub_ec']);assert math.isfinite(labels[rid])
    assert set(inputs)==set(labels)==train
    records={}
    for rid in sorted(train):
        farm,day,hour=rid.split('_');node=(farm,int(day));h=int(hour)
        records.setdefault(node,{})[h]={**inputs[rid],'sub_ec':labels[rid]}
    scales={f['farm']:f['weather_scales_train_only'] for f in audit['farms']}
    model=PrefixSourceCH2(layout,records,graph['links'],scales)
    assert len(records)==audit['train_records'] and len(graph['links'])==audit['links_checked']
    assert len(model.cyclic_roots)==len(graph['cycles'])
    payload={'layout_sha256':sha(HERE/'BLK_layout_v2.json'),'source_sha256':pins,
             'records':[{'farm':f,'day':d,'hours':{str(h):dict(obs) for h,obs in hours.items()}}
                        for (f,d),hours in sorted(model.records.items())],
             'links':graph['links'],'weather_scales':scales}
    encoded=json.dumps(payload,ensure_ascii=False,sort_keys=True,allow_nan=False)
    # JSON round trip must reconstruct exactly the immutable training state.
    decoded=json.loads(encoded)
    replay_records={(v['farm'],v['day']):{int(h):obs for h,obs in v['hours'].items()} for v in decoded['records']}
    replay=PrefixSourceCH2(layout,replay_records,decoded['links'],decoded['weather_scales'])
    assert replay.root_by_node==model.root_by_node and replay.cyclic_roots==model.cyclic_roots
    assert replay.bounds==model.bounds and replay.supported_columns==model.supported_columns
    assert all(dict(replay.records[node][h])==dict(model.records[node][h]) for node in records for h in range(24))
    assert all(sha(p)==value for p,value in pins.items())
    output=HERE/'CH2_training_reference_v1.json';assert not output.exists()
    output.write_text(encoded,encoding='utf-8')
    result={'status':'TRAIN_REFERENCE_PREPARATION_PASS_NOT_QUERY_OR_PERFORMANCE_GATE',
            'source_sha256':pins,'reference_file_sha256':sha(output),'reference_payload_digest':digest(payload),
            'train_rows':len(train),'train_records':len(records),'links':len(graph['links']),
            'cyclic_components':len(model.cyclic_roots),'query_rows_numerically_parsed':0,
            'gap_rows_numerically_parsed':0,'heldout_labels_numerically_parsed':0,
            'JSON_roundtrip_exact':True,'GPU_used':False,'model_fit':False,'performance_evaluated':False,
            'candidate_execution_registered':False,
            'limits':['Original label-informed graph preserved, not a physical-source truth',
                      'Training input and labels only; CSV/hash still read full strings/bytes',
                      'No query selection, blend or score was run; three-candidate statistics/lineage registration still needed',
                      'Original validator graphs must be rebuilt from each permissible training set']}
    receipt=HERE/'CH2_training_reference_preparation_v1.json';assert not receipt.exists()
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','train_rows','train_records','links','cyclic_components','JSON_roundtrip_exact']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
