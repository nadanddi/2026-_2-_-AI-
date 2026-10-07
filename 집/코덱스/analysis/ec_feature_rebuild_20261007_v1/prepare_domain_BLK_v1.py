"""Prepare the fixed domain24 family map and audit prefix features, no fitting/scoring."""
from blk_baseline_data_v1 import *
import csv, json, math, platform, time
from domain_features_v1 import build_domain, dynamics, FAMILIES
from checkpoint_v1 import atomic

def main():
    started=time.monotonic()
    gate_path=HERE/'BLK_verified_baseline_receipt_v4.json'
    gate=json.loads(gate_path.read_text(encoding='utf-8'))
    assert gate['whole_pipeline_gate_passed']
    assert all(sha(Path(p))==value for p,value in gate['transitive_source_sha256'].items())
    sgsource=ROOT/'집/클로드/submission14_ec_sg2/sg2post.py'
    sgaudit=json.loads((HERE/'BLK_SG2_refonly_audit_v2.json').read_text(encoding='utf-8'))
    assert sha(sgsource)==sgaudit['source_sha256']
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    ctx=BLKContext(layout)
    observations={**ctx.reference_inputs,**ctx._query}
    frame=dataframe(observations)
    features,selected,metadata=build_domain(frame)
    assert len(features)==6960 and set(features.index)==ctx.train_ids|ctx.query_ids
    assert not set(features.index)&ctx.gap_ids
    with (HERE/'feature_candidates_v4.csv').open(encoding='utf-8-sig',newline='') as handle:
        candidates=[row for row in csv.DictReader(handle) if row['stage']=='1']
    assert {row['candidate_id'] for row in candidates}==set(FAMILIES)
    assert all(row['previous_catalog'] and row['difference_from_previous'] for row in candidates)
    # Every registered window/lag is included as one family; no best-window search.
    checks=0
    for block in layout['blocks']:
        for day in [block['query_days'][0],block['query_days'][-1]]:
            prefix=f'{block["farm"]}_{day:03d}_'
            dayframe=frame[frame.row_id.str.startswith(prefix)]
            hours=dayframe.row_id.str[-2:].astype(int)
            for hour in [0,1,6,12,23]:
                query_id=f'{prefix}{hour:02d}'
                short,ss,_=build_domain(dayframe[hours<=hour])
                np.testing.assert_allclose(short.loc[query_id],features.loc[query_id],atol=0,rtol=0,equal_nan=True)
                assert ss==selected
                poison=dayframe.copy()
                poison.loc[hours>hour,[c for c in frame if c!='row_id']]=77777.
                poisoned,_,_=build_domain(poison)
                np.testing.assert_allclose(poisoned.loc[query_id],features.loc[query_id],atol=0,rtol=0,equal_nan=True)
                checks+=2
    # Independent scalar rolling/lag computation for the three newly expanded families.
    scalar_checks=0
    for block in layout['blocks']:
        day=block['query_days'][0]
        prefix=f'{block["farm"]}_{day:03d}_'
        for origin in ['co2_dose_response','co2_uptake_proxy','sealed_run']:
            series={h:float(features.loc[f'{prefix}{h:02d}',origin]) for h in range(24)}
            for hour in range(24):
                rid=f'{prefix}{hour:02d}'
                for window in [1,2,3,4,6]:
                    vals=[series[j] for j in range(max(0,hour-window+1),hour+1) if math.isfinite(series[j])]
                    total=math.fsum(vals)
                    mean=total/len(vals) if vals else float('nan')
                    expected={'count':len(vals),'sum':total if vals else float('nan'),'mean':mean,
                              'std':math.sqrt(math.fsum((v-mean)**2 for v in vals)/len(vals)) if vals else float('nan')}
                    for operator,value in expected.items():
                        actual=float(features.loc[rid,f'{origin}__{operator}{window}'])
                        assert (math.isnan(actual) and math.isnan(value)) or abs(actual-value)<1e-9
                        scalar_checks+=1
    shuffled,_,_=build_domain(frame.sample(frac=1,random_state=47))
    np.testing.assert_allclose(shuffled,features,rtol=0,atol=0,equal_nan=True)
    other=frame.copy()
    other.loc[other.row_id.str.startswith('F47_'),[c for c in frame if c!='row_id']]=33333.
    changed,_,_=build_domain(other)
    f13=[r for r in features.index if r.startswith('F13_')]
    np.testing.assert_allclose(changed.loc[f13],features.loc[f13],rtol=0,atol=0,equal_nan=True)
    # Reset after a missing hour and at midnight is tested by the underlying feature audit;
    # each new family is also recomputed on a record with an absent hour.
    rid=sorted(ctx.query_ids)[0]
    prefix=rid[:8]
    example=frame[frame.row_id.str.startswith(prefix)].copy()
    missing=example[~example.row_id.str.endswith('_03')]
    missingfeatures,_,_=build_domain(missing)
    for origin in ['co2_dose_response','co2_uptake_proxy','sealed_run']:
        row=f'{prefix}04'
        assert math.isnan(float(missingfeatures.loc[row,origin+'__lag1']))
    sources=dict(gate['transitive_source_sha256'])
    files=['domain_features_v1.py','prepare_domain_BLK_v1.py','fast_features_v2.py','causal_features_v1.py',
           'feature_candidates_v4.csv','preregistration_v2.json','blk_r3_baseline_v1.py',
           'critique_BLK_scored_diagnostics_v1.md','BLK_verified_baseline_receipt_v4.json',
           'BLK_diagnostic_results_v1.json']
    for name in files:sources[str((HERE/name).resolve())]=sha(HERE/name)
    sources[str(sgsource.resolve())]=sha(sgsource)
    for path in (ROOT/'집/클로드/research/domain_2026-10-03').iterdir():
        if path.is_file():sources[str(path.resolve())]=sha(path)
    mapping={row['candidate_id']:{'family':row['family'],'source':row['source'],
             'previous_catalog':row['previous_catalog'],'difference':row['difference_from_previous'],
             'feature_columns':selected[row['candidate_id']],
             'additional_ET_columns':[c for c in selected[row['candidate_id']] if c not in FULL_R3]}
             for row in candidates}
    assert all(value['additional_ET_columns'] for value in mapping.values())
    result={'status':'DOMAIN24_FEATURE_PREPARATION_PASS_NOT_FIT_REGISTERED',
        'baseline':'CPU reference-only cached recipe, historical GPU/submission identity not claimed',
        'layout_sha256':sha(HERE/'BLK_layout_v2.json'),'gate_sha256':sha(gate_path),
        'candidate_cap':24,'seeds':[47,1414,6464],'target_member':'ET only; all other members and postprocess fixed',
        'family_map':mapping,'columns':len(features.columns),'rows':len(features),
        'prefix_future_checks':checks,'independent_special_rolling_checks':scalar_checks,
        'shuffle_and_other_farm_checks':2,'missing_hour_checks':3,
        'train_ids':sorted(ctx.train_ids),'query_ids':sorted(ctx.query_ids),
        'source_sha256':sources,'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        'performance_evaluated':False,'heldout_truth_loaded':False,'model_fit':False,'GPU_used':False,
        'next':['independent family map/causality review','register runner/scorer source before fit',
                'raw baseline ET replay then72 candidate fits with frozen data/ID hashes',
                'full mixed postprocess audit before any candidate score'],
        'limits':['Same BLK labels already used; exploration only, no confirmation',
                  'BLK is not promoted to main validator; TM/P2LOO/EL1 checks remain mandatory',
                  'Domain family effects conditional on ET and fixed baseline; not all model/feature interactions',
                  'CH2 source-matching bounded followup remains untested'],
        'elapsed_seconds':time.monotonic()-started}
    out=HERE/'DOMAIN24_BLK_preparation_v1.json';assert not out.exists()
    atomic(out,json.dumps(result,ensure_ascii=False,allow_nan=False))
    print(f'Domain24 preparation PASS: {checks} prefix/future, {scalar_checks} independent rolling checks; no fit or score',flush=True)

if __name__=='__main__':main()
