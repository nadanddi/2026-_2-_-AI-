"""Independent csv/math.fsum audit of posthoc concentration, no fitting or tuning."""
from pathlib import Path
import csv,json,math,datetime,hashlib
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3];D=H/'risk_diagnosis_v1';L=ROOT/'연구실/코덱스/local'/H.name
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def mean(v):return math.fsum(v)/len(v)
def same(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-11,rel_tol=1e-11),(a,b)
rows=read(L/'stage34_rows_v2.csv');official=json.loads((D/'completion.json').read_text(encoding='utf-8'));screen=json.loads((H/'stage34_results_v2/completion.json').read_text(encoding='utf-8'));assert hashlib.sha256((L/'stage34_rows_v2.csv').read_bytes()).hexdigest()==screen['row_output_sha'];group=defaultdict(list)
for r in rows:
    for c in ['seed','day','hour']:r[c]=int(r[c])
    for c in ['A','candidate','sub_ec','delta','risk']:r[c]=float(r[c])
    r['loss']=(r['candidate']-r['sub_ec'])**2-(r['A']-r['sub_ec'])**2;group[r['seed'],r['farm'],r['day']].append(r)
changed=[r for r in rows if r['delta']>0];assert len(changed)==official['modified_row_occurrences']==57;assert len({r['row_id'] for r in changed})==official['unique_modified_time_rows']==22;assert len({(r['farm'],r['day']) for r in changed})==official['unique_modified_days']==4;assert sum(r['day']>=179 for r in rows)==official['pass2_query_rows']==0
modified=read(D/'modified_rows.csv');assert {(r['row_id'],int(r['seed'])) for r in modified}=={(r['row_id'],r['seed']) for r in changed}
for r in modified:same(r['sse_change'],next(u['loss'] for u in changed if u['row_id']==r['row_id'] and u['seed']==int(r['seed'])))
concentration=[]
for seed in [7,101,2024]:
    q=[r for r in rows if r['seed']==seed];total=math.fsum(r['loss'] for r in q);case=math.fsum(r['loss'] for r in q if r['farm']=='F47' and r['day']==161);daily=[math.fsum(u['loss'] for u in v) for (s,f,d),v in group.items() if s==seed];expected=dict(seed=seed,total_sse_change=total,F47_161_sse_change=case,case_fraction_of_net_change=case/total,other_days_sse_change=total-case,improved_days=sum(x<0 for x in daily),worsened_days=sum(x>0 for x in daily),unchanged_days=sum(x==0 for x in daily));ref=next(r for r in official['loss_breakdown'] if r['seed']==seed)
    for c,v in expected.items():same(v,ref[c])
    concentration.append(expected)
selected=read(D/'selected_original_cases.csv')
for r in selected:
    key=(int(r['seed']),r['farm'],int(r['day']));q=group.get(key,[]);assert len(q)==int(r['rows']);same(sum(u['delta']>0 for u in q),r['modified_rows']);same(math.fsum(u['loss'] for u in q),r['sse_change'])
    if q:
        for c,v in dict(ymean=mean([u['sub_ec'] for u in q]),pmean=mean([u['A'] for u in q]),cmean=mean([u['candidate'] for u in q]),risk_max=max(u['risk'] for u in q)).items():same(v,r[c])
    else:assert all(r[c]=='' for c in ['ymean','pmean','cmean','risk_max'])
assert len(selected)==12
highmods=[r for r in changed if r['high']=='True'];assert len(highmods)==len(official['high_modified'])==1;actual=highmods[0];ref=official['high_modified'][0];assert actual['row_id']=='F13_177_00' and actual['seed']==101
for c in ['A','candidate','sub_ec','delta','risk']:same(actual[c],ref[c])
same(actual['loss'],ref['sse_change']);assert actual['A']<actual['sub_ec'] and actual['candidate']<actual['A'] and actual['loss']>0
out=dict(status='PASS_POSTHOC_DIAGNOSIS_ARITHMETIC_ONLY',candidate_status='SCREEN_REJECT_UNCHANGED',modified_seed_rows=57,unique_modified_time_rows=22,modified_unique_days=4,pass2_rows=0,concentration=concentration,high_harm=dict(row_id=actual['row_id'],seed=101,target=actual['sub_ec'],baseline=actual['A'],candidate=actual['candidate'],delta=actual['delta'],sse_increase=actual['loss']),original_case98_status='ABSENT_FROM_PARTIAL_QUERY',new_fit=0,new_candidate=0,limit='posthoc selected-day concentration diagnostic; no retuning or new efficacy claim')
path=H/('verify_risk_diagnosis_v2_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
