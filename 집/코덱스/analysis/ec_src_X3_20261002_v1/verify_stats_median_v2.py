"""중앙값 통제 통계의 독립 수동 rank/pinv/fsum 재계산."""
import verify_stats_v1 as v
from collections import defaultdict
from statistics import median
import csv,math,json,hashlib
import numpy as np
HERE=v.HERE

def main():
    days=list(csv.DictReader((HERE/'day_features.csv').open(encoding='utf-8-sig',newline='')))
    stats=list(csv.DictReader((HERE/'stats_median_v2.csv').open(encoding='utf-8-sig',newline='')))
    mags=defaultdict(list)
    for r in csv.DictReader((HERE/'row_features.csv').open(encoding='utf-8-sig',newline='')):
        mags[(r['farm'],int(r['day']))].append(float(r['magnitude']))
    strata=sorted({r['farm']+'_'+r['section'] for r in days});reports=[]
    for s in stats:
        m,o=s['metric'],s['outcome'];xx=[float(r[m]) for r in days];yy=[float(r[o]) for r in days]
        if len(set(xx))<=1 or len(set(yy))<=1:
            assert s['status']=='UNTESTABLE_CONSTANT' and s['found']=='False'
            reports.append(dict(metric=m,outcome=o,status='UNTESTABLE_CONSTANT'));continue
        design=[]
        for r in days:
            key=r['farm']+'_'+r['section'];day=float(r['day'])/250
            row=[int(key==k) for k in strata]
            for k in strata:row.extend([day*int(key==k),day*day*int(key==k)])
            row.append(median(mags[(r['farm'],int(r['day']))]))
            if o!='ec_mean':
                lv=math.log(max(float(r['ec_mean']),1e-9));row.extend([lv,lv*lv])
            design.append(row)
        z=np.asarray(design,float);xr=np.asarray(v.rank(xx));yr=np.asarray(v.rank(yy))
        resx=xr-z@(np.linalg.pinv(z)@xr);resy=yr-z@(np.linalg.pinv(z)@yr)
        rho=v.corr(resx.tolist(),resy.tolist());diff=abs(rho-float(s['rho_adjusted']))
        assert diff<1e-9,(m,o,rho,s['rho_adjusted'])
        reports.append(dict(metric=m,outcome=o,status='PASS',rho_adjusted=rho,absolute_difference=diff))
    output=dict(status='PASS',method='원문행수치 csv + statistics.median + 수동 tied-rank + pinv + math.fsum',comparisons=reports,
        final_lock_scored=False,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'verify_stats_median_v2.py',HERE/'stats_median_v2.csv',HERE/'row_features.csv']})
    (HERE/'verification_stats_median_v2.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
