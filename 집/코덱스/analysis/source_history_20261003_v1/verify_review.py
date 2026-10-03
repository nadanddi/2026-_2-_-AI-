"""Independent public-only result audit. Never execute other AI label loaders."""
from pathlib import Path
import sys,json,math,csv
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd,numpy as np
from scipy.stats import spearmanr
CHECKS=[]
def ck(n,b):
    assert b,n
    CHECKS.append(n)
def independent(g,c,y='y'):
    return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(g[y],g[c]))/len(g))
def main():
    OUT=ROOT/'집/코덱스/local/source_history_20261003_v1'
    o=pd.read_csv(OUT/'oof.csv',float_precision='round_trip')
    scores=pd.read_csv(HERE/'scores_v1.csv',float_precision='round_trip')
    for r in scores.itertuples():
        g=o[(o.validator==r.validator)&(o.seed==r.seed)]
        for c in ['baseline','candidate']:ck('new_RMSE_'+c,abs(independent(g,c)-getattr(r,c))<1e-12)
    oldrows=[]
    # Filter strings before conversion; no EL1 score or reserved source file.
    with open(ROOT/'집/클로드/research/local/ec3_ST8_all.csv',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['validator'] in ['DIAG10','A','B','EXT10','EXT12']:oldrows.append(r)
    old=pd.DataFrame(oldrows)
    numeric=['day','hour','sub_ec','pB']+[f'{c}_{s}' for c in ['r3s','st'] for s in [7,101,2024]]
    old[numeric]=old[numeric].astype(float)
    review=[]
    for v,g in old.groupby('validator'):
        for s in [7,101,2024]:
            rb=independent(g,f'r3s_{s}','sub_ec');rc=independent(g,f'st_{s}','sub_ec')
            ck('ST8_numpy',abs(rb-np.sqrt(np.mean((g.sub_ec-g[f'r3s_{s}'])**2)))<1e-12)
            review.append(dict(validator=v,seed=s,n=len(g),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1)))
    pd.DataFrame(review).to_csv(HERE/'ST8_independent_scores_v1.csv',index=False)
    # Independent complete-weather-pair enumeration from public OOF membership only.
    raw=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id','out_temp','out_hum','out_rad','out_wspd','in_temp'])
    raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[8:10].astype(int)
    public=o[(o.validator=='DIAG10')&(o.seed==7)]
    keys=set(zip(public.farm,public.day));daily=public.assign(e=public.y-public.baseline).groupby(['farm','day']).agg(y=('y','mean'),e=('e','mean'))
    weather={k:g.sort_values('hour')[['out_temp','out_hum','out_rad','out_wspd']].to_numpy() for k,g in raw.groupby(['farm','day']) if k in keys}
    pairs=[]
    for k,a in weather.items():
        nxt=(k[0],k[1]+1)
        if nxt in weather and np.isfinite(a).all() and np.array_equal(a,weather[nxt]):pairs.append((k,nxt))
    result=json.loads((HERE/'result_v1.json').read_text(encoding='utf-8'))
    for rec in result['structure']:ck('pair_enumeration',rec['public_complete_pairs']==sum(a[0]==rec['farm'] for a,b in pairs))
    # Recompute pair-only persistence with public labels; not the other AI full-label population.
    persistence=[]
    for farm in ['F13','F47']:
        for role in [0,1]:
            seq=sorted([pair[role] for pair in pairs if pair[role][0]==farm and pair[role][1]<179],key=lambda x:x[1])
            x=[];y=[];ex=[];ey=[]
            for cur in seq:
                prev=[q for q in seq if 0<cur[1]-q[1]<=10]
                if prev:
                    q=prev[-1];x.append(daily.loc[q,'y']);y.append(daily.loc[cur,'y']);ex.append(daily.loc[q,'e']);ey.append(daily.loc[cur,'e'])
            rho=spearmanr(x,y).statistic if len(x)>5 else None
            erho=spearmanr(ex,ey).statistic if len(x)>5 else None
            if len(x)>5:ck('rank_correlation',abs(rho-pd.Series(x).rank().corr(pd.Series(y).rank()))<1e-12)
            persistence.append(dict(farm=farm,role=role,n=len(x),level_rho=rho,residual_rho=erho))
    # Scalar prefix check, independent of implementation groupby.
    f,d=next(iter(keys));g=raw[(raw.farm==f)&(raw.day==d)&(raw.hour<=6)].sort_values('hour')
    vals=g.in_temp.dropna().to_list()
    if vals:ck('prefix_scalar',abs(math.fsum(vals)/len(vals)-g.in_temp.mean())<1e-12)
    segments=[]
    for s in [7,101,2024]:
        g=o[(o.validator=='DIAG10')&(o.seed==s)].copy()
        hi=g.groupby(['farm','day']).y.transform('mean')>=1
        for name,m in [('high',hi),('late',g.day>=179),('history',g.history_n>0)]:
            z=g[m];segments.append(dict(seed=s,segment=name,n=len(z),days=len(z[['farm','day']].drop_duplicates()),baseline=independent(z,'baseline'),candidate=independent(z,'candidate'),baseline_bias=float((z.baseline-z.y).mean()),candidate_bias=float((z.candidate-z.y).mean())))
    pd.DataFrame(segments).to_csv(HERE/'segments_v1.csv',index=False)
    # Independent replay of each day correction from nested residual records.
    for (v,k),g0 in o[o.seed==7].groupby(['validator','fold']):
        g=g0.sort_values(['farm','day','hour'])
        ck('past_only',all(q==-1 or q<d for q,d in zip(g.last_day,g.day)))
        ck('missing_history_no_change',np.array_equal(g.loc[g.history_n==0,'baseline'].to_numpy(),g.loc[g.history_n==0,'candidate'].to_numpy()))
    evidence=dict(status='PASS',checks=len(CHECKS),names=CHECKS,persistence_public_only=persistence,ST8_public_rows=len(old),ST8_query_pB_exact_one_share=float((old.pB==1).mean()),history_DIAG_row_share=float((public.history_n>0).mean()),decision=result['decision'],limitations=['paired role is a proxy, not verified physical identity','SE2 full-history diagnostic not independent nested validation','inner model has fewer training days and sparse residual history','No EL1 or final lock scored; no new submission'])
    (HERE/'verification_v1.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
