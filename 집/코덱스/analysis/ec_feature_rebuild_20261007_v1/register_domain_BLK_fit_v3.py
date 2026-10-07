"""Freeze exact domain24 families and exploratory decisions before any domain fit."""
import csv,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
prep=read('DOMAIN24_BLK_preparation_v2.json')
events=read('DOMAIN24_event_audit_v1.json')
assert prep['status']=='DOMAIN24_FEATURE_PREPARATION_PASS_NOT_FIT_REGISTERED'
assert prep['prefix_future_checks']==160 and prep['independent_special_rolling_checks']==17856
assert events['status']=='PASS' and events['independent_current_lag_rate_d2_rolling_scalar_checks']==1792
assert events['features_code_sha256']==sha(HERE/'domain_features_v2.py')
assert events['code_sha256']==sha(HERE/'audit_domain_events_v1.py')
assert all(sha(path)==value for path,value in prep['source_sha256'].items())
with (HERE/'feature_candidates_v4.csv').open(encoding='utf-8-sig',newline='') as handle:
    reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
descriptions={
    'D05':'현재 습도차 흐름과 환기·포그 값전환 직전 관측 대비 변화; 동일기록 prefix',
    'D09':'포그×VPD 흐름 및 포그 값전환 직전 관측 대비 VPD 변화; 이후 입력 미사용',
    'D11':'공급중 ΔCO2 흐름과 0→양수/양수→0 전환시각 ΔCO2 분리; 생리 인과효과 아님',
    'D13':'다일 평균 제외·동일날 1/2/3/4/6h 변화량·고정 전체창 묶음',
    'D22':'포그 자체 상태전환·연속시간 및 값전환 직전 관측 대비 온도·습도·VPD 변화',
}
for row in rows:
    if row['stage']!='1':continue
    if row['candidate_id'] in descriptions:row['difference_from_previous']=descriptions[row['candidate_id']]
    row['status']='REGISTERED_DOMAIN24_CPU_RAW_FIT_PENDING'
    params=json.loads(row['parameters'])
    params.update({'exact_columns':prep['family_map'][row['candidate_id']]['feature_columns'],
                   'actual_additional_ET_columns':prep['family_map'][row['candidate_id']]['additional_ET_columns'],
                   'same_name_baseline_columns_removed':True,'window1_aliases_kept_consistently':True,
                   'new_actuator_event_reference':'most recent observed value-change t-1 environmental value; missing/gap resets; h0 NaN',
                   'CO2_onset_offset':'binary >0 / <=0 transitions, same observed one-hour CO2 change',
                   'performance_scope':'BLK exploration only; all24 still require original validators'})
    row['parameters']=json.dumps(params,ensure_ascii=False)
out=HERE/'feature_candidates_v5.csv';assert not out.exists()
with out.open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
sources=dict(prep['source_sha256'])
for name in ['DOMAIN24_BLK_preparation_v2.json','DOMAIN24_event_audit_v1.json','audit_domain_events_v1.py',
             'feature_candidates_v5.csv','register_domain_BLK_fit_v3.py','run_domain_BLK_raw_v3.py',
             'critique_DOMAIN24_preparation_v1.md','critique_DOMAIN24_preparation_v2.md']:
    sources[str((HERE/name).resolve())]=sha(HERE/name)
reg={'status':'REGISTERED_BEFORE_DOMAIN_FIT','runner_sha256':sha(HERE/'run_domain_BLK_raw_v3.py'),
     'preparation_sha256':sha(HERE/'DOMAIN24_BLK_preparation_v2.json'),'family_map':prep['family_map'],
     'candidate_cap':24,'seeds':[47,1414,6464],'target_member':'ET only, all other members and postprocess fixed',
     'ET_parameters':{'n_estimators':600,'max_features':1.,'min_samples_leaf':1,'n_jobs_fit':4,'n_jobs_predict':1,
                      'imputer':'train-only median, sklearn default drops all-missing train columns'},
     'raw_fit_count':72,'baseline_ET_replays':3,'numeric_audit_atol':1e-6,'rtol':0,
     'alias_policy':'retain full grammar consistently; remove identical baseline names only; log deterministic mean1/sum1 aliases',
     'known_family_overlap':'D09 event VPD and D22 event VPD are identical input series and dynamics; not independent findings',
     'source_sha256':sources,'baseline_gate_sha256':prep['gate_sha256'],
     'statistics':{'scopes':['BLK_QUERY_ROLE','BLK_RAW_PASS'],'variant_count':48,'previous_BLK_variants':6,
                   'cumulative_current_BLK_variant_count':54,'comparison_alpha':.025/54,
                   'seedmean_loss':'row mean over3seed of candidate squared error minus baseline squared error',
                   'draws':20000,'rng_seed':2026100702,
                   'sampling':'4blocks replacement per farm; sum block loss / sum rows',
                   'p':'(1+count(delta>=0))/20001; ties worse',
                   'screen':'all3seed overall RMSE smaller and p<.025/54; diagnostics only',
                   'normal_high':'score-only full24hour dailymean EC>=1',
                   'position':'same frozen floor(3*day_index/block_length) front/middle/back',
                   'CI':'individual descriptive95 linear percentiles; not multiplicity adjusted'},
     'original_validators':'All24 candidates TM/P2LOO/EL1 x3seeds remain; no exclusion based on BLK result',
     'adoption_permitted':False,'final_confirmation':'unchanged first-unused seed/layout once, all cells improve, .025/finalist count',
     'core_review':'DF01 source mismatch closed by actual event implementation in v2; other review gaps repaired in audit/runner3',
     'heldout_truth_loaded_for_domain_fit':False,'GPU_used':False,
     'limits':['BLK labels exposed earlier, exploration only; historical GPU/submission identity not claimed',
               '54 adjustment covers current BLK comparisons only, not entire history or all196 candidates',
               'Family score includes alias/split-sampling effects, not pure new information',
               'raw audit does not replace final mixed postprocess causality gate',
               'CH2 source-matching remains separate untested bounded followup']}
assert set(reg['family_map'])=={f'D{i:02d}' for i in range(1,25)}
out=HERE/'DOMAIN24_BLK_fit_registration_v3.json';assert not out.exists()
out.write_text(json.dumps(reg,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('Domain24 exact families/source/ET/statistics registered before first fit',flush=True)
