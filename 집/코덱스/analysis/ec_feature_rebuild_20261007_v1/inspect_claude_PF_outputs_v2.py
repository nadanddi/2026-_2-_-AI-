"""Read-only PF1/PF2 prediction metadata and lineage audit; never parse target values."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib,math
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CLAUDE=ROOT/'집/클로드/research'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    registry=read(HERE/'original_validator_endpoint_registry_v1.json')
    expected={(f['validator'],f['fold'],rid) for f in registry['folds'] for rid in f['ordered_query_ids']}
    assert len(expected)==10848
    tm={tuple(v) for v in read(HERE/'fold_registry_v1.json')['TM_days']}
    paths=[CLAUDE/'local/ec3_PF1_all.csv',CLAUDE/'local/ec3_PF2_all.csv',
           CLAUDE/'ec3_PF1_tabpfn_upgrade_v1.py',CLAUDE/'ec3_PF2_anchor_difference_features_v1.py',
           CLAUDE/'ec3_AF0_anchor_features_v1.py',CLAUDE/'ec3_SG2_reference_knn_level_v1.py',
           CLAUDE/'ec2_DC4_exact_twin_anchor_v1.py',CLAUDE/'env_extra_gpu.py',
           HERE/'original_validator_endpoint_registry_v1.json',HERE/'fold_registry_v1.json',Path(__file__)]
    hashes={str(p.resolve()):sha(p) for p in paths}
    frames={};info=[]
    for name in ['PF1','PF2']:
        path=CLAUDE/f'local/ec3_{name}_all.csv';table={};counts=Counter();folds=set();tm_rows=0
        columns=['pfnA','pfnB','pfnC','pfnD']+(['pfnE'] if name=='PF2' else [])
        members=[f'{m}_{s}' for s in [47,1414,6464] for m in ['et','lgb','mlp']]
        with path.open(encoding='utf-8-sig',newline='') as handle:
            reader=csv.DictReader(handle);header=reader.fieldnames
            assert 'sub_ec' in header and all(c in header for c in columns+members+['validator','validation_fold','row_id'])
            for row in reader:
                rid=row['row_id'];validator=row['validator'];fold=int(row['validation_fold'])
                key=validator,fold,rid
                assert key not in table and key in expected
                farm,day,hour=rid.split('_');day=int(day);hour=int(hour)
                assert row['farm']==farm and int(row['day'])==day and int(row['hour'])==hour
                # Only predictions and reference-derived clipping bounds are interpreted numerically.
                values={c:float(row[c]) for c in columns+members+['lo','hi']}
                assert all(math.isfinite(v) for v in values.values()) and values['lo']<=values['hi']
                table[key]=values;counts[validator]+=1;folds.add((validator,fold))
                tm_rows+=validator=='DIAG10' and (farm,day) in tm
        assert set(table)<=expected
        missing=sorted(expected-set(table))
        complete=set(table)==expected
        if complete:
            assert len(folds)==66 and tm_rows==2664
            assert dict(counts)=={'DIAG10':8640,'EL1':1104,'P2LOO':1104}
        frames[name]=table
        info.append({'name':name,'rows':len(table),'folds':len(folds),'validator_rows':dict(counts),
            'TM_rows':tm_rows,'complete_expected_metadata':complete,'missing_expected_rows':len(missing),'missing_expected_keys':missing,'prediction_columns':columns,'target_values_converted':0,'metadata_and_finite_predictions_pass':True})
    shared=['pfnA','pfnB','pfnC','pfnD','lo','hi']+[f'{m}_{s}' for s in [47,1414,6464] for m in ['et','lgb','mlp']]
    reused={c:max(abs(frames['PF1'][key][c]-frames['PF2'][key][c]) for key in set(frames['PF1'])&set(frames['PF2'])) for c in shared}
    assert max(reused.values())<1e-12
    assert all(sha(path)==value for path,value in hashes.items())
    result={'status':'PF1_PF2_METADATA_AVAILABLE_NOT_CAUSAL_GATE_OR_SCORE',
        'source_sha256':hashes,'outputs':info,'shared_keys_compared':len(set(frames['PF1'])&set(frames['PF2'])),'shared_prediction_bound_max_differences':reused,
        'targets_numeric_conversion':0,'model_fit':False,'GPU_used':False,'score_recomputed':False,'adoption_permitted':False,
        'lineage':{'PF1_D':'FULL38 plus7 AF0 anchor summaries, search over general training reference records',
            'PF2_E':'PF1_D plus7 differences in observed 0..h means relative to an anchor',
            'proposed_CH2':'Label-informed graph plus same-block fixedleft3/right3 input signature; no general-reference search'},
        'source_review_pending':['Research SG2 preparation globally standardizes pass1 weather before fold exclusions',
            'PF reference uses labset minus validation rather than exact purged train IDs',
            'Default TabPFN fit/predict cache policy and execution environment fingerprints need strict audit',
            'Comparison blend source has no final SG2 call, so it is not the registered cached SG2 baseline'],
        'limits':['Complete-looking CSV metadata cannot prove legal feature generation or model completion receipts',
                  'CSV reader and file hash read target strings/bytes but no target values were numerically interpreted or scored',
                  'No PF1/PF2 performance conclusion; no duplicate model or generic neighbor fit run']}
    out=HERE/'CLAUDE_PF_outputs_metadata_v2.json';assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'outputs':[{k:v for k,v in item.items() if k!='missing_expected_keys'} for item in info],'shared_keys_compared':result['shared_keys_compared'],'shared_max_difference':max(reused.values()),'target_scoring':False},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
