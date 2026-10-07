"""Freeze six comparisons before any CH2 real-query prediction; no query values read."""
from pathlib import Path
import json,csv,hashlib,ast
from ch2_reference_loader_v1 import load_reference
from run_ch2_BLK_v2 import baseline_preflight,SCOPES,SEEDS
from ch2_prefix_sources_v3 import METHODS

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    model,layout,refdetails=load_reference()
    baseline,gate=baseline_preflight()
    paths={Path(p).resolve() for p in gate['transitive_source_sha256']}
    paths.update(Path(p).resolve() for p in refdetails['direct_and_transitive_source_sha256'])
    for name in ['run_ch2_BLK_v2.py','register_ch2_BLK_v2.py','ch2_query_context_v1.py','ch2_reference_loader_v1.py',
                 'CH2_training_reference_preparation_v1.json','CH2_training_reference_file_replay_v1.json',
                 'CH2_training_reference_v1.json','BLK_verified_baseline_receipt_v4.json',
                 'critique_CH2_prefix_sources_v2.md','critique_CH2_training_reference_v1.md',
                 'critique_CH2_reference_loader_v1.md','CLAUDE_PF_outputs_metadata_v2.json','feature_candidates_v5.csv']:
        paths.add((HERE/name).resolve())
    for name in ['run_ch2_BLK_v2.py','ch2_query_context_v1.py']:ast.parse((HERE/name).read_text(encoding='utf-8'))
    rows=[]
    with (HERE/'feature_candidates_v5.csv').open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    assert len(rows)==196 and len({r['candidate_id'] for r in rows})==196
    for scope in SCOPES:
        for method in METHODS:
            row={f:'' for f in fields}
            row.update(candidate_id=f'CH2_{scope}_{method}',stage='0',family=method,
                       source='사용자 추가 최우선 BLK 양끝·사슬',previous_catalog='6.346;6.348;6.375;6.379;6.380',
                       difference_from_previous='원 slope-corrected Hungarian graph·고정 앞뒤3flank 입력 배정·순환성분 fallback. 기존 mutual endpoint guard와 다름',
                       status='REGISTERED_CH2_INFERENCE_NOT_SCORED',performance='UNTESTED')
            # Preserve the actual existing CSV schema; formula details belong to new registration.
            if 'parameters' in fields:row['parameters']=json.dumps({'registration':'CH2_BLK_registration_v2.json','scope':scope,'method':method},ensure_ascii=False)
            rows.append(row)
    assert len(rows)==202 and len({r['candidate_id'] for r in rows})==202
    csvout=HERE/'feature_candidates_v7.csv';assert not csvout.exists()
    with csvout.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    paths.add(csvout.resolve())
    reg={'status':'REGISTERED_BEFORE_REAL_QUERY_INFERENCE','methods':list(METHODS),'scopes':list(SCOPES),'seeds':list(SEEDS),
         'source_sha256':{str(p):sha(p) for p in sorted(paths,key=str)},'ordered_query_ids':baseline['row_ids'],
         'layout_sha256':sha(HERE/'BLK_layout_v2.json'),'primary_scope':'BLK_QUERY_ROLE','diagnostic_scope':'BLK_RAW_PASS',
         'comparison_count':6,'previous_BLK_comparisons':54,'cumulative_BLK_comparisons':60,
         'baseline':'Already whole-verified CPU reference-only cached recipe; no new model fit',
         'baseline_limitation':'Not byte identity with historical GPU/submission14; BLK pass1-only',
         'methods_exact':{'CH2_REFONLY_GUARD':'Unique nearest input-only component from fixed6flanks; nearest calendar left/right must both belong; cyclic component abstains',
                          'FLANK_SOURCE_MATCH':'Unique nearest input-only component from fixed6flanks; nearest compatible left/right anchors in that component; cyclic/missing/tie/one-sided abstain',
                          'PAST_QUERY_PREFIX_STATE':'FLANK_SOURCE_MATCH plus current-component == nearest component from same-block complete earlier query days and current prefix; disagreements abstain'},
         'ranking':'Per observed query day average11 common-reference input terms; templates mean within each side/component then equal side mean. Current-only rank uses current0..h, history rank uses all same-block past plus current0..h. No EC term in ranking',
         'metric':'weather weighted squared train-hourdiff-SD; indoor .25scaled squared; controls .25absdelta/50. Exact prefix3 constants pinned',
         'numerical_tie_tolerance':1e-12,'no_query_fit':True,'no_physical_chronology_inferred':True,
         'reference_graph':'Original training-only slope-corrected EC/input cost Hungarian assignment, not physical source truth; whole cyclic components abstain',
         'reference_scope':'Layout train IDs only; exact fixed3left+3right search templates, both public endpoints allowed',
         'prefix_policy':'Exact same-block all preceding query day24h plus current0..h; metadata/schema precheck before numeric conversion. Other blocks excluded',
         'pipeline':['reused final baseline with single shrink/clip/SG2/clip already applied',
                     'if supported .8baseline+.2linear EC endpoint interpolation using relative recordhour',
                     'train-reference EC bounds clip; otherwise exact baseline'],
         'statistics':{'loss':'per-row mean_seed squared-error delta, not squared error of ensemble mean',
                       'bootstrap':'Python3.12 random.Random seed2026100702; farm-stratified4blocks replacement per farm; row-weighted',
                       'draws':20000,'rng_seed':2026100702,'p_worse':'(1+count(delta>=0))/20001; ties worse',
                       'comparison_alpha':.025/60,'CI':'descriptive individual95%; cumulative multiplicity adjusted interval1-.05/60',
                       'screen':'all3seeds whole BLK RMSE improve and p_worse<.025/60; never adoption alone'},
         'original_validators_required':['all3seeds TM111','P2LOO','EL1; graph/scale freshly rebuilt on permitted fold training rows'],
         'final_confirmation':'First-unused seeds/layout once, separately frozen before confirmation; not yet registered',
         'prior_trials_difference':'6.375 mutualbest/.05/.005 guard had zero coverage; here original CH2 graph and fixed flank input signatures; no tuned thresholds/weights',
         'PF_lineage':'PF1/PF2 general training-neighbor features in PFN vs fixed6flank graph-conditioned endpoint blend. PF1 full/PF2 partial snapshot metadata only, no causal/performance claim; no duplicated GPU/generic neighbor fit',
         'normal_high_split':'Score-only heldout daily ECmean>=1; inaccessible to inference',
         'position':'floor3*i/n within query block; score-only reporting',
         'scoring_permitted':False,'stage':'Registration+runner review pending; actual inference and independent gate required before scorer',
         'adoption_permitted':False,'GPU_used':False,'query_values_read_during_registration':False}
    out=HERE/'CH2_BLK_registration_v2.json';assert not out.exists()
    out.write_text(json.dumps(reg,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'CH2 six comparisons registered; cumulative60; {len(paths)} sources; CSV202 candidates; no query inference/score',flush=True)

if __name__=='__main__':main()
