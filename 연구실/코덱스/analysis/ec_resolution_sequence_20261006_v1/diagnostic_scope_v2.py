"""Post-hoc coverage and selected-case concentration; never changes the registered decision."""
from pathlib import Path
import csv,json,math,hashlib
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
P=ROOT/'연구실/코덱스/local'/H.name/'stage34_rows_v2.csv'
with P.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
def metric(q):
    a=math.sqrt(math.fsum((float(r['A'])-float(r['sub_ec']))**2 for r in q)/len(q))
    b=math.sqrt(math.fsum((float(r['candidate'])-float(r['sub_ec']))**2 for r in q)/len(q))
    return dict(rows=len(q),days=len({(r['farm'],r['day']) for r in q}),baseline=a,candidate=b,change_pct=(b/a-1)*100)
coverage=[];excluded=[]
for seed in ['7','101','2024']:
    q=[r for r in rows if r['seed']==seed];ds={(r['farm'],int(r['day'])) for r in q}
    coverage.append(dict(seed=int(seed),all_days=len(ds),pass1_days=sum(d<179 for f,d in ds),pass2_days=sum(d>=179 for f,d in ds),pass2_status='UNTESTED_ZERO_QUERY_ROWS',F13_98_status='ABSENT_FROM_PARTIAL_OUTER_QUERY'))
    rem=[r for r in q if not (r['farm']=='F47' and int(r['day'])==161)]
    excluded.append(dict(seed=int(seed),excluded_case='F47_161',posthoc=True,all_remaining=metric(rem),ordinary_remaining=metric([r for r in rem if r['high']=='False'])))
result=dict(status='COMPLETE_POSTHOC_SCOPE_DIAGNOSIS',coverage=coverage,selected_case_excluded=excluded,source_rows_sha=hashlib.sha256(P.read_bytes()).hexdigest(),registered_decision_unchanged='SCREEN_REJECT',candidate_tuning=False,new_fit=0)
dest=H/'risk_diagnosis_scope_v2.json'
with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(excluded,indent=2))
