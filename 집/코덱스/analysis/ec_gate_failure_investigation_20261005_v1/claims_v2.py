"""Fresh report numbers with stdlib CSV/math crosschecks; no model selection."""
from pathlib import Path
import csv,json,math
H=Path(__file__).resolve().parent
def read(n):
    with (H/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    v=json.loads((H/'verification_v2.json').read_text(encoding='utf-8'));assert v['status'].startswith('PASS_DIAGNOSIS_AND_META270')
    pooled=read('pooled_v2.csv');folds=read('fold_summaries_v1.csv');days=read('daily_v1.csv');records=[];daily=[]
    for mode in ['LR','LGB','MLP']:
        for seed in [7,101,2024]:
            for scope in ['outer_actual','inner_resub','inner_meta_oof','thin_B_fixed_g','thin_x_fixed_B','thin_B_thin_g','no_down','no_up']:
                for segment in ['all','high','ordinary','pass2']:
                    z=[r for r in pooled if r['mode']==mode and r['validator']=='DIAG10' and int(r['seed'])==seed and r['scope']==scope and r['segment']==segment];assert len(z)==1;r=z[0]
                    ff=[q for q in folds if all(q[k]==r[k] for k in ['mode','validator','seed','scope','segment'])]
                    sums={c:math.fsum(float(q[c]) for q in ff) for c in ['sse_A','sse_P','net','positive_loss','wrong_positive_loss','overshoot_positive_loss','partial_oracle_gain','total_oracle_gain']}
                    for c,s in sums.items():assert abs(s-float(r[c]))<1e-8
                    change=100*(math.sqrt(sums['sse_P']/sums['sse_A'])-1);wrong=sums['wrong_positive_loss']/sums['positive_loss'] if sums['positive_loss'] else 0;partial=sums['partial_oracle_gain']/sums['total_oracle_gain'] if sums['total_oracle_gain'] else 0
                    assert abs(change-float(r['change_pct']))<1e-10 and abs(wrong-float(r['wrong_share_positive']))<1e-12 and abs(partial-float(r['partial_share_oracle_gain']))<1e-12
                    records.append(dict(mode=mode,seed=seed,scope=scope,segment=segment,n=int(r['n']),change_pct=change,wrong_share_positive=wrong,partial_share_oracle_gain=partial,net=sums['net'],gate_mean=float(r['gate_mean']),residual_mean=float(r['residual_mean']),delta_mean=float(r['delta_mean'])))
            dd=[r for r in days if r['mode']==mode and r['validator']=='DIAG10' and int(r['seed'])==seed];assert len(dd)==360
            net=math.fsum(float(r['net']) for r in dd);level=math.fsum(float(r['level']) for r in dd);shape=math.fsum(float(r['shape']) for r in dd);assert abs(net-level-shape)<1e-8
            pos=sorted([max(0,float(r['net'])) for r in dd],reverse=True);positive=math.fsum(pos)
            high=math.fsum(float(r['net']) for r in dd if float(r['mean_y'])>=1)
            daily.append(dict(mode=mode,seed=seed,net=net,level=level,shape=shape,level_fraction_net=level/net,high_net=high,top5_positive_daily_share=math.fsum(pos[:5])/positive,top10_positive_daily_share=math.fsum(pos[:10])/positive))
    geo=json.loads((H/'geometry_v2.json').read_text(encoding='utf-8'));gd=read('geometry_days_v2.csv');bounds=[]
    for s in [7,101,2024]:
        hi=[r for r in gd if int(r['seed'])==s and r['high']=='True'];assert len(hi)==31
        bounds.append(dict(seed=s,high_days=31,unreachable_high_days=sum(r['unreachable_above_day_level']=='True' for r in hi)))
    matched=json.loads((H/'matched_summary_v3.json').read_text(encoding='utf-8'))
    result=dict(status='PASS_FRESH_REPORT_NUMBERS_STD_LIB',records=records,daily=daily,geometry=geo['records'],bounds=bounds,matched=[r for r in matched['results'] if r['mode']=='LR' and r['segment'] in ['all','high','pass2']],crosschecks=['RMSE from fold SSE math.fsum','wrong-loss fraction numerator denominator','partial oracle benefit numerator denominator','daily level plus shape equals net'])
    with (H/'claims_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('REPORT_CLAIMS_PASS')
if __name__=='__main__':main()

