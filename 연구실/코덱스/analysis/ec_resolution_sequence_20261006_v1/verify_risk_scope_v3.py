from pathlib import Path
import csv,json,math,hashlib,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[3];P=ROOT/'연구실/코덱스/local'/H.name/'stage34_rows_v2.csv';proof=json.loads((H/'risk_diagnosis_scope_v2.json').read_text(encoding='utf-8'))
with P.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert hashlib.sha256(P.read_bytes()).hexdigest()==proof['source_rows_sha']
def metric(q):
    before=math.sqrt(math.fsum((float(r['A'])-float(r['sub_ec']))**2 for r in q)/len(q));after=math.sqrt(math.fsum((float(r['candidate'])-float(r['sub_ec']))**2 for r in q)/len(q));return dict(rows=len(q),days=len({(r['farm'],r['day']) for r in q}),baseline=before,candidate=after,change_pct=(after/before-1)*100)
checked=[]
for seed in [7,101,2024]:
    q=[r for r in rows if int(r['seed'])==seed];days={(r['farm'],int(r['day'])) for r in q};coverage=next(r for r in proof['coverage'] if r['seed']==seed);assert len(days)==coverage['all_days']==coverage['pass1_days']==141 and coverage['pass2_days']==0;assert ('F13',98) not in days and all(day<179 for farm,day in days)
    rem=[r for r in q if not (r['farm']=='F47' and int(r['day'])==161)];ref=next(r for r in proof['selected_case_excluded'] if r['seed']==seed);current={}
    for label,qq in [('all_remaining',rem),('ordinary_remaining',[r for r in rem if r['high']=='False'])]:
        m=metric(qq)
        for c,v in m.items():assert math.isclose(v,ref[label][c],abs_tol=1e-12,rel_tol=1e-12),(seed,label,c,v,ref[label][c])
        current[label]=m
    checked.append(dict(seed=seed,**current))
high_day=[r for r in rows if r['seed']=='101' and r['farm']=='F13' and r['day']=='177'];assert len(high_day)==24;high_day_y=math.fsum(float(r['sub_ec']) for r in high_day)/24;assert high_day_y>=1
out=dict(status='PASS_SCOPE_AND_SELECTED_CASE_SENSITIVITY',pass2='UNTESTED_ZERO_ROWS',F13_98='ABSENT',registered_decision='SCREEN_REJECT_UNCHANGED',sensitivity=checked,F13_177_day_mean_ec=high_day_y,fit=0,tuning=False,scope='posthoc loss concentration, not a new acceptance rule')
path=H/('verify_risk_scope_v3_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
