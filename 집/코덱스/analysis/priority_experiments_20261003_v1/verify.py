"""Independent numerical, split, prefix and reference-model audit."""
import sys,math,json,hashlib
sys.dont_write_bytecode=True
from run import *
from collections import defaultdict

def main():
    checks=[]
    def ok(name,cond,**details):
        if not cond:raise AssertionError((name,details))
        checks.append(dict(name=name,passed=True,**details))
    score=pd.read_csv(HERE/'scores_v1.csv',float_precision='round_trip')
    for arm in ARMS:
        source=OUT/'corrected_et_v2' if arm in ('EC_RARE_ET','EC_H0CHANGE') else OUT
        paths=sorted(source.glob(f'{arm}_*.csv'))
        d=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths])
        ok(arm+'_complete_folds',len(paths)==(22 if arm.startswith('EC_') else 12))
        for _,s in score[score.arm==arm].iterrows():
            a=d[(d.validator==s.validator)&(d.seed==s.seed)&(d.context==s.context)]
            for col,expected in [('baseline',s.baseline_rmse),('candidate',s.candidate_rmse)]:
                manual=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(a[col],a.y))/len(a))
                ok(arm+'_'+s.validator+'_'+str(s.seed)+'_'+s.context+'_'+col,abs(manual-expected)<1e-12,rmse=manual)
            ok('score_denominators',len(a)==s.rows and len(set(zip(a.farm,a.day)))==s.days)
        if arm in ('EC_DAY20','EC_RARE_ET','EC_H0CHANGE'):
            a=d[(d.validator=='DIAG10')&(d.seed==7)];blocks=defaultdict(list)
            for r in a.itertuples():blocks[(r.farm,int(r.day)//5)].append((r.candidate-r.y)**2-(r.baseline-r.y)**2)
            keys=sorted(blocks);delta=np.array([math.fsum(blocks[key]) for key in keys]);count=np.array([len(blocks[key]) for key in keys]);rng=np.random.default_rng(20261003);ix=[]
            for farm in ('F13','F47'):
                ids=np.array([i for i,key in enumerate(keys) if key[0]==farm]);ix.append(ids[rng.integers(0,len(ids),(20000,len(ids)))])
            ix=np.concatenate(ix,axis=1);samples=delta[ix].sum(axis=1)/count[ix].sum(axis=1);p=float(np.mean(samples>=0));ci=np.quantile(samples,[.025/K,1-.025/K]);expected=score[(score.arm==arm)&(score.validator=='DIAG10')&(score.seed==7)].iloc[0]
            ok('independent_block_bootstrap_'+arm,p==expected.p_worse and abs(ci[0]-expected.ci_low)<1e-12 and abs(ci[1]-expected.ci_high)<1e-12)
    ec,core,wv,folds,outer=S.loadec()
    for name,k,tm,vm in folds:
        tr,va=ec[tm],ec[vm];t=set(zip(tr.farm,tr.day));v=set(zip(va.farm,va.day))
        ok('EC_'+name+'_'+str(k)+'_buffer',all((f,d+delta) not in t for f,d in v for delta in (-1,0,1)))
    tables=prefix_tables(ec,core.RAW)
    # Independent scalar raw-row sums and future-input mutation.
    for farm in ('F13','F47'):
        day=int(ec.loc[ec.farm==farm,'day'].iloc[0])
        for c in CUTS:
            raw=ec[(ec.farm==farm)&(ec.day==day)&(ec.hour<=c)].sort_values('hour')
            for col in core.RAW:
                vals=[float(x) for x in raw[col] if pd.notna(x)]
                a=tables[c].loc[(farm,day),col+'_mean']
                manual=math.fsum(vals)/len(vals) if vals else float('nan')
                ok('scalar_prefix_mean',bool(np.isnan(a) and np.isnan(manual)) or abs(a-manual)<1e-10)
                h0=raw.loc[raw.hour==0,col].iloc[0];end=raw.loc[raw.hour==c,col].iloc[0]
                for suffix,v in [('h0',h0),('current',end),('delta',end-h0)]:
                    actual=tables[c].loc[(farm,day),col+'_'+suffix]
                    ok('scalar_prefix_'+suffix,bool(pd.isna(actual) and pd.isna(v)) or actual==v)
            changed=ec.copy();mask=(changed.farm!=farm)|((changed.farm==farm)&((changed.day>day)|((changed.day==day)&(changed.hour>c))))
            changed.loc[mask,core.RAW]=12345.
            other=prefix_tables(changed,core.RAW)[c].loc[(farm,day)]
            ok('future_other_farm_invariance',other.equals(tables[c].loc[(farm,day)]))
    # Fresh unweighted original ET, using the same training/season inputs, versus old cache.
    name,k,tm,vm=folds[0];tr,va=S.seasonal(ec[tm],ec[vm],wv)
    cols=[c for c in core.FULL if c!='day']+['season']
    with threadpool_limits(limits=2):p=core.predict_model(core.et(7),tr,va,cols)
    old=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip')
    old=old[(old.validator==name)&(old.validation_fold==k)].set_index('row_id').etS_7.reindex(va.row_id).to_numpy()
    diff=float(np.max(np.abs(p-old)));ok('original_ET_fresh_reproduction',diff<1e-10,maxdiff=diff)
    temp,_,_,_,tfolds,touter=S.loadtemp()
    for name,k,fd in tfolds:
        tm,vm=S.common.split_mask(temp,fd);t=set(zip(temp.loc[tm,'farm'],temp.loc[tm,'day']));v=set(zip(temp.loc[vm,'farm'],temp.loc[vm,'day']))
        ok('TEMP_'+name+'_'+str(k)+'_buffer',all((f,d+delta) not in t for f,d in v for delta in (-1,0,1)))
    # Compare the frozen initial W30G membership with stored incidence criteria, independently.
    daily=pd.read_csv(HERE/'W40G_daily_audit.csv',float_precision='round_trip');q=daily[(daily.validator=='DIAG10')&(daily.seed==7)&(daily.context=='1-8')]
    fail=[(r.farm,int(r.day)) for r in q.itertuples() if r.air_complete and abs(r.gap)>=2 and r.old_rmse>.5]
    result=json.loads((HERE/'result_v1.json').read_text(encoding='utf-8'))
    ok('W40G_fail_count',len(fail)==result['W40G_old_failure_days']==16)
    still=sum(r.new_rmse>.5 for r in q.itertuples() if (r.farm,int(r.day)) in fail)
    ok('W40G_still_fail_count',still==result['W40G_old_failures_still_fail'])
    # Classification AUC independent pairwise ranking on DIAG10, four horizons per target.
    rr=pd.read_csv(HERE/'risk_scores_v1.csv',float_precision='round_trip')
    for target,threshold in [('EC',.1),('TEMP',.5)]:
        d=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in OUT.glob(f'RISK_{target}_DIAG10_*.csv')])
        squared=defaultdict(list)
        for r in d.itertuples():squared[(r.farm,int(r.day))].append((r.baseline-r.y)**2)
        isfail={key:math.sqrt(math.fsum(vals)/len(vals))>threshold for key,vals in squared.items()}
        for h in (0,6,12,23):
            a=d[d.hour==h];positive=[];negative=[]
            for r in a.itertuples():(positive if isfail[(r.farm,int(r.day))] else negative).append(r.risk)
            manual=math.fsum(1. if p>n else .5 if p==n else 0. for p in positive for n in negative)/(len(positive)*len(negative))
            expected=rr[(rr.target==target)&(rr.validator=='DIAG10')&(rr.hour==h)].auc.iloc[0]
            ok('risk_auc_pairwise',abs(manual-expected)<1e-12)
    jsonout(HERE/'verification_v1.json',dict(status='PASS',checks=checks,count=len(checks),baseline_ET_maxdiff=diff,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},limitations=['Public OOF reused across prior studies','No new lock scoring','EL1 guard required if public criteria pass','Temperature contexts reuse fixed PFN; new PFN target training not tested']))
    print('VERIFY_PASS',len(checks),'ET_maxdiff',diff,flush=True)
if __name__=='__main__':main()
