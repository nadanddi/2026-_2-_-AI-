from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));import stage2_v1 as R
import numpy as np,pandas as pd
q=pd.read_csv(R.S.L/'stage34_rows_v2.csv',float_precision='round_trip');result=json.loads((H/'stage34_results_v2/completion.json').read_text())
assert R.sha(R.S.L/'stage34_rows_v2.csv')==result['row_output_sha']
D=H/'risk_diagnosis_v1';D.mkdir(exist_ok=False)
q['sse_change']=(q.candidate-q.sub_ec)**2-(q.A-q.sub_ec)**2
changed=q[q.delta>0].copy();changed.to_csv(D/'modified_rows.csv',index=False)
rows=[]
for seed,g in q.groupby('seed'):
    for farm,day in [('F47',160),('F47',161),('F13',98),('F13',112)]:
        a=g[(g.farm==farm)&(g.day==day)]
        rows.append(dict(seed=int(seed),farm=farm,day=day,rows=len(a),ymean=float(a.sub_ec.mean()) if len(a) else None,pmean=float(a.A.mean()) if len(a) else None,cmean=float(a.candidate.mean()) if len(a) else None,risk_max=float(a.risk.max()) if len(a) else None,modified_rows=int((a.delta>0).sum()),sse_change=math.fsum(a.sse_change)))
pd.DataFrame(rows).to_csv(D/'selected_original_cases.csv',index=False)
loss=[]
for seed,g in q.groupby('seed'):
    daily=g.groupby(['farm','day']).sse_change.sum();total=math.fsum(g.sse_change);case=math.fsum(g.loc[(g.farm=='F47')&(g.day==161),'sse_change'])
    loss.append(dict(seed=int(seed),total_sse_change=total,F47_161_sse_change=case,case_fraction_of_net_change=case/total,other_days_sse_change=total-case,improved_days=int((daily<0).sum()),worsened_days=int((daily>0).sum()),unchanged_days=int((daily==0).sum())))
save=dict(status='COMPLETE_POSTHOC_DIAGNOSIS_ONLY',unique_modified_time_rows=int(changed.row_id.nunique()),modified_row_occurrences=len(changed),unique_modified_days=len(changed[['farm','day']].drop_duplicates()),pass2_query_rows=int((q.day>=179).sum()),loss_breakdown=loss,high_modified=changed[changed.high][['row_id','seed','sub_ec','A','candidate','delta','risk','smooth_lgb','sse_change']].to_dict('records'),new_fit=0,new_candidates=0)
R.write(D/'completion.json',save);print(json.dumps(save,indent=2),flush=True)
