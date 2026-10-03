from pathlib import Path
import sys,json,math,hashlib
sys.dont_write_bytecode=True
from run import HERE,OUT,ARMS,K
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss

def scores(d):
    rows=[]
    for keys,q in d.groupby(['target','arm','validator','seed','context'],sort=True):
        y=q.y.to_numpy();a=q.baseline.to_numpy();b=q.candidate.to_numpy();e0=(a-y)**2;e1=(b-y)**2
        rm0=float(np.sqrt(e0.mean()));rm1=float(np.sqrt(e1.mean()))
        folds=q.assign(e=e1).groupby('fold').e.mean()**.5
        r=dict(zip(['target','arm','validator','seed','context'],keys));r.update(rows=len(q),days=len(q[['farm','day']].drop_duplicates()),baseline_rmse=rm0,candidate_rmse=rm1,delta_pct=100*(rm1/rm0-1),fold_mean=float(folds.mean()),fold_sd=float(folds.std()) if len(folds)>1 else None)
        if keys[2]=='DIAG10':
            g=q.assign(n=1,block5=q.day//5,delta=e1-e0).groupby(['farm','block5']).agg(n=('n','sum'),delta=('delta','sum'))
            rng=np.random.default_rng(20261003);ix=[]
            for f in ('F13','F47'):
                ids=np.flatnonzero(g.index.get_level_values(0)==f)
                ix.append(ids[rng.integers(0,len(ids),(20000,len(ids)))])
            ix=np.concatenate(ix,axis=1);samples=g.delta.to_numpy()[ix].sum(1)/g.n.to_numpy()[ix].sum(1)
            r.update(p_worse=float((samples>=0).mean()),ci_low=float(np.quantile(samples,.025/K)),ci_high=float(np.quantile(samples,1-.025/K)),alpha=.025/K,blocks=len(g))
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    frames=[]
    for arm in ARMS:
        source=OUT/'corrected_et_v2' if arm in ('EC_RARE_ET','EC_H0CHANGE') else OUT
        assert (source/f'{arm}_audit.json').exists(),('Incomplete',arm)
        fs=[pd.read_csv(p,float_precision='round_trip') for p in sorted(source.glob(f'{arm}_*.csv'))]
        frames.extend(fs)
    d=pd.concat(frames,ignore_index=True)
    assert not d.duplicated(['arm','validator','fold','seed','context','row_id']).any()
    s=scores(d);s.to_csv(HERE/'scores_v1.csv',index=False)
    decisions=[]
    for arm in ARMS:
        a=s[s.arm==arm];dg=a[a.validator=='DIAG10']
        direction=bool((a.delta_pct<0).all());ci=bool((dg.p_worse<.025/K).all() and (dg.ci_high<0).all())
        decisions.append(dict(arm=arm,score_cells=len(a),same_direction=direction,bootstrap_pass=ci,status='PUBLIC_PASS_NEEDS_EL1' if direction and ci else 'REJECT'))
    pd.DataFrame(decisions).to_csv(HERE/'decisions_v1.csv',index=False)
    seg=[]
    for keys,q in d[d.validator=='DIAG10'].groupby(['target','arm','seed','context']):
        yy=q.groupby(['farm','day']).y.mean();high=dict(yy>=1)
        for tag,mask in [('early',q.day<179),('late',q.day>=179),('F13',q.farm=='F13'),('F47',q.farm=='F47'),('high_EC',np.array([high[(f,day)] for f,day in zip(q.farm,q.day)]) if keys[0]=='EC' else np.zeros(len(q),bool))]:
            a=q.loc[mask]
            if not len(a):continue
            seg.append(dict(target=keys[0],arm=keys[1],seed=keys[2],context=keys[3],segment=tag,rows=len(a),days=len(a[['farm','day']].drop_duplicates()),baseline_rmse=float(np.sqrt(((a.baseline-a.y)**2).mean())),candidate_rmse=float(np.sqrt(((a.candidate-a.y)**2).mean())),baseline_bias=float((a.baseline-a.y).mean()),candidate_bias=float((a.candidate-a.y).mean())))
    pd.DataFrame(seg).to_csv(HERE/'segments_v1.csv',index=False)
    base=pd.read_csv(OUT/'baseline_update_oof.csv',float_precision='round_trip');scores(base).to_csv(HERE/'W40G_scores_v1.csv',index=False)
    daily=pd.read_csv(HERE/'W40G_daily_audit.csv',float_precision='round_trip');q=daily[(daily.validator=='DIAG10')&(daily.seed==7)&(daily.context=='1-8')]
    oldfail=q.air_complete & (q.gap.abs()>=2) & (q.old_rmse>.5)
    jsonresult=dict(decisions=decisions,family_k=K,score_cells=len(s),W40G_old_failure_days=int(oldfail.sum()),W40G_old_failures_still_fail=int((q.loc[oldfail].new_rmse>.5).sum()),W40G_old_failures_old_rmse=float(np.sqrt(np.mean(q.loc[oldfail].old_rmse**2))),W40G_old_failures_new_rmse=float(np.sqrt(np.mean(q.loc[oldfail].new_rmse**2))),leakage_scope='stored public EC labels only, MASK training, causal within-day features',new_submission=False)
    risks=[]
    for target,th in [('EC',.1),('TEMP',.5)]:
        assert (OUT/'risk_audit.json').exists()
        rr=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted(OUT.glob(f'RISK_{target}_*.csv'))])
        for name,a in rr.groupby('validator'):
            yd=a.assign(e=(a.baseline-a.y)**2).groupby(['fold','farm','day']).e.mean()**.5
            for h in (0,6,12,23):
                b=a[a.hour==h].set_index(['fold','farm','day']);y=(yd.reindex(b.index)>th).astype(int);p=b.risk.to_numpy();low=p<=np.quantile(p,.25);high=p>=np.quantile(p,.75)
                risks.append(dict(target=target,validator=name,hour=h,days=len(y),failure_days=int(y.sum()),auc=float(roc_auc_score(y,p)) if y.nunique()==2 else None,brier=float(brier_score_loss(y,p)),constant_brier=float(np.mean((float(y.mean())-y.to_numpy())**2)),low_risk_days=int(low.sum()),high_risk_days=int(high.sum()),low_risk_rmse=float(np.sqrt(np.mean(yd.reindex(b.index).to_numpy()[low]**2))),high_risk_rmse=float(np.sqrt(np.mean(yd.reindex(b.index).to_numpy()[high]**2)))))
    pd.DataFrame(risks).to_csv(HERE/'risk_scores_v1.csv',index=False)
    (HERE/'result_v1.json').write_text(json.dumps(jsonresult,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(decisions).to_string(index=False));print(s.groupby('arm').delta_pct.agg(['min','max']).to_string());print('W40G failures',jsonresult['W40G_old_failure_days'],jsonresult['W40G_old_failures_still_fail'])
if __name__=='__main__':main()
