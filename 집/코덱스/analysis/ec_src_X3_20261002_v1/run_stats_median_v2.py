"""사전문서와 일치하는 magnitude 중앙값 통제 통계; 기존 파일 보존."""
import run as b
import numpy as np
import pandas as pd
import math,json
from scipy.stats import rankdata

def main():
    daily=pd.read_csv(b.HERE/'day_features.csv')
    rows=pd.read_csv(b.HERE/'row_features.csv',usecols=['farm','day','magnitude'])
    median=rows.groupby(['farm','day']).magnitude.median().rename('magnitude_median').reset_index()
    daily=daily.merge(median,on=['farm','day'],validate='one_to_one')
    # b.residualize expects the existing column name; only this in-memory copy uses its median.
    daily['magnitude_mean']=daily.magnitude_median
    out=[]
    for m in b.METRICS:
        for yname in b.OUTCOMES:
            d=daily[np.isfinite(daily[m])&np.isfinite(daily[yname])].reset_index(drop=True)
            rawconstant=d[m].nunique()<=1 or d[yname].nunique()<=1
            if rawconstant:
                rho=p=lo=hi=np.nan;x=y=None;farm_rho={'F13':np.nan,'F47':np.nan}
            else:
                x=b.residualize(d,rankdata(d[m]),yname);y=b.residualize(d,rankdata(d[yname]),yname)
                rho,p,lo,hi=b.block_stats(d,x,y)
                farm_rho={f:b.correlation(x[d.farm.eq(f)],y[d.farm.eq(f)]) for f in ['F13','F47']}
            counts={k:int(d[m+'_group'].eq(k).sum()) for k in [0,1]}
            farm_counts={f:{k:int(((d.farm==f)&(d[m+'_group']==k)).sum()) for k in [0,1]} for f in ['F13','F47']}
            enough=all(n>=15 for n in counts.values()) and all(n>=5 for z in farm_counts.values() for n in z.values())
            consistent=all(math.isfinite(v) and v*rho>0 for v in farm_rho.values()) if math.isfinite(rho) else False
            out.append(dict(metric=m,outcome=yname,n=len(d),status='UNTESTABLE_CONSTANT' if rawconstant else 'TESTED',
                rho_adjusted=rho,p_block=p,p_bonferroni=min(1,p*15) if math.isfinite(p) else np.nan,
                ci_lower=lo,ci_upper=hi,group0_days=counts[0],group1_days=counts[1],enough_groups=enough,
                farm_F13_rho=farm_rho['F13'],farm_F47_rho=farm_rho['F47'],farm_sign_consistent=consistent,
                found=bool(enough and consistent and math.isfinite(p) and p*15<.01)))
    pd.DataFrame(out).to_csv(b.HERE/'stats_median_v2.csv',index=False,encoding='utf-8-sig')
    median.to_csv(b.HERE/'day_magnitude_median_v2.csv',index=False,encoding='utf-8-sig')
    summary=dict(magnitude_control='daily_median',planned_comparisons=15,
        tested_comparisons=sum(z['status']=='TESTED' for z in out),
        untestable_constant=sum(z['status']=='UNTESTABLE_CONSTANT' for z in out),
        found_comparisons=[z for z in out if z['found']],
        source_hashes={p.name:b.sha(p) for p in [b.HERE/'run_stats_median_v2.py',b.HERE/'EXECUTION_NOTE_v2.md',b.HERE/'row_features.csv',b.HERE/'day_features.csv']},
        final_lock_scored=False,original_files_modified=False)
    b.dump(b.HERE/'results_stats_median_v2.json',b.to_json(summary))
    print(json.dumps(b.to_json(summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
