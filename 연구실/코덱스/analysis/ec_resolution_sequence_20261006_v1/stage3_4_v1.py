from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));import stage2_v1 as R
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from risk_core_v2 import downward
C=R.CONFIG
def eligible(q):return (q.act_vent_tdz>=C['vent_zero'])&(q.act_circfan_tdm<C['fan_mean'])
def metric(q):
    if len(q)==0:return dict(rows=0,days=0,baseline=None,candidate=None,change_pct=None,mse_delta=None)
    a=q.A-q.sub_ec;b=q.candidate-q.sub_ec;before=float(np.sqrt(np.mean(a*a)));after=float(np.sqrt(np.mean(b*b)))
    return dict(rows=len(q),days=len(q[['farm','day']].drop_duplicates()),baseline=before,candidate=after,change_pct=(after/before-1)*100,mse_delta=float(np.mean(b*b-a*a)))
def risk_metrics(q):
    target=(q.A-q.sub_ec>C['label_threshold']).to_numpy();flag=(q.risk>C['threshold']).to_numpy();changed=(q.A-q.candidate>0).to_numpy()
    def ratios(sel):
        tp=int(np.sum(target&sel));return dict(selected=int(sel.sum()),true_positive=tp,precision=tp/int(sel.sum()) if sel.sum() else None,recall=tp/int(target.sum()) if target.sum() else None)
    return dict(positive_rows=int(target.sum()),auc=float(roc_auc_score(target,q.risk)) if len(np.unique(target))==2 else None,risk_flag=ratios(flag),modified=ratios(changed))
def main():
    receipt=json.loads((H/'stage2_receipt_v1.json').read_text(encoding='utf-8'));assert receipt['status']=='COMPLETE_PARTIAL_RISK_FIT'
    O=R.S.L/'risk_stage2_v1';assert not (O/'worker.lock').exists();source=O/'risk_rows.csv';assert R.sha(source)==receipt['output_sha']
    q=pd.read_csv(source,float_precision='round_trip');assert len(q)==10152
    q['eligible']=eligible(q);q['candidate']=downward(q.A,q.smooth_lgb,q.risk,q.eligible,threshold=C['threshold'],cap=C['cap'])
    q['delta']=q.A-q.candidate;q['y_day']=q.groupby(['seed','farm','day']).sub_ec.transform('mean');q['high']=q.y_day.ge(1)
    assert q.groupby(['seed','farm','day']).size().eq(24).all()
    D=H/'stage34_results_v1';D.mkdir(exist_ok=False)
    results=[];risks=[];reason=[]
    for seed,g in q.groupby('seed'):
        subsets={'all':g,'ordinary':g[~g.high],'high':g[g.high],'pass1':g[g.day<179],'pass2':g[g.day>=179]}
        subsets.update({'farm_'+f:g[g.farm==f] for f in ['F13','F47']});subsets.update({'fold_'+str(k):g[g.k==k] for k in range(4)})
        stats={n:metric(v) for n,v in subsets.items()}
        for name,m in stats.items():results.append(dict(seed=int(seed),subset=name,**m))
        for name in ['all','ordinary']:
            if not stats[name]['rows'] or not stats[name]['candidate']<stats[name]['baseline']:reason.append(f'{seed}:{name}:no_strict_improvement')
        if not stats['high']['rows'] or stats['high']['candidate']>stats['high']['baseline']+1e-12:reason.append(f'{seed}:high:protection_failed')
        if stats['pass2']['rows'] and stats['pass2']['change_pct']>=2:reason.append(f'{seed}:pass2:guard_failed')
        risks.append(dict(seed=int(seed),**risk_metrics(g),eligible_rows=int(g.eligible.sum()),modified_high_rows=int(((g.delta>0)&g.high).sum())))
    daily=[]
    for (seed,farm,day),g in q.groupby(['seed','farm','day']):daily.append(dict(seed=int(seed),farm=farm,day=int(day),ymean=float(g.sub_ec.mean()),pmean=float(g.A.mean()),cmean=float(g.candidate.mean()),modified_rows=int((g.delta>0).sum()),risk_max=float(g.risk.max()),**metric(g)))
    av=q.groupby(['row_id','farm','day','hour'],as_index=False)[['sub_ec','A','candidate','risk']].mean();av['y_day']=av.groupby(['farm','day']).sub_ec.transform('mean');av['high']=av.y_day.ge(1)
    mean_scores={n:metric(v) for n,v in {'all':av,'ordinary':av[~av.high],'high':av[av.high]}.items()}
    dest=R.S.L/'stage34_rows_v1.csv';assert not dest.exists();q.to_csv(dest,index=False)
    pd.DataFrame(results).to_csv(D/'subset_scores.csv',index=False);pd.DataFrame(daily).to_csv(D/'daily_scores.csv',index=False)
    result=dict(status='SCREEN_REJECT' if reason else 'SCREEN_PASS_NEEDS_FULL',reasons=reason,seed_scores=results,seed_mean=mean_scores,risk_metrics=risks,rows=len(q),unique_days=len(av[['farm','day']].drop_duplicates()),modified_rows=int((q.delta>0).sum()),modified_unique_days=len(q[q.delta>0][['farm','day']].drop_duplicates()),source_sha=R.sha(__file__),protocol_sha=R.sha(H/'PROTOCOL2_v1.md'),stage2_receipt_sha=R.sha(H/'stage2_receipt_v1.json'),row_output_sha=R.sha(dest),full80=False,adoption=False,bootstrap=False)
    R.write(D/'completion.json',result)
    print(json.dumps(dict(status=result['status'],reasons=reason,seed_mean=mean_scores,risk_metrics=risks,modified_rows=result['modified_rows'],modified_unique_days=result['modified_unique_days']),indent=2),flush=True)
if __name__=='__main__':main()
