"""저장 통계의 독립 rank/pinv/math.fsum 검산. 상수지표는 비교불능."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
sys.dont_write_bytecode=True
import csv,json,math,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def rank(xs):
    order=sorted(range(len(xs)),key=lambda j:xs[j]);out=[0.0]*len(xs);i=0
    while i<len(xs):
        k=i+1
        while k<len(xs) and xs[order[k]]==xs[order[i]]:k+=1
        avg=((i+1)+k)/2
        for j in range(i,k):out[order[j]]=avg
        i=k
    return out

def corr(a,b):
    am=math.fsum(a)/len(a);bm=math.fsum(b)/len(b)
    cov=math.fsum((x-am)*(y-bm) for x,y in zip(a,b))
    aa=math.fsum((x-am)**2 for x in a);bb=math.fsum((y-bm)**2 for y in b)
    return cov/math.sqrt(aa*bb)

def main():
    days=list(csv.DictReader((HERE/'day_features.csv').open(encoding='utf-8-sig',newline='')))
    stats=list(csv.DictReader((HERE/'stats.csv').open(encoding='utf-8-sig',newline='')))
    strata=sorted({r['farm']+'_'+r['section'] for r in days})
    reports=[]
    for s in stats:
        m,o=s['metric'],s['outcome'];xx=[float(r[m]) for r in days];yy=[float(r[o]) for r in days]
        if len(set(xx))<=1 or len(set(yy))<=1:
            reports.append(dict(metric=m,outcome=o,status='UNTESTABLE_CONSTANT',group1_days=int(s['group1_days']),original_found=s['found']))
            assert s['found']=='False'
            continue
        design=[]
        for r in days:
            key=r['farm']+'_'+r['section'];day=float(r['day'])/250
            row=[int(key==k) for k in strata]
            for k in strata:row.extend([day*int(key==k),day*day*int(key==k)])
            row.append(float(r['magnitude_mean']))
            if o!='ec_mean':
                lv=math.log(max(float(r['ec_mean']),1e-9));row.extend([lv,lv*lv])
            design.append(row)
        z=np.asarray(design,float);xr=np.asarray(rank(xx));yr=np.asarray(rank(yy))
        resx=xr-z@(np.linalg.pinv(z)@xr);resy=yr-z@(np.linalg.pinv(z)@yr)
        rho=corr(resx.tolist(),resy.tolist());raw=corr(xr.tolist(),yr.tolist())
        diff=abs(rho-float(s['rho_adjusted']));rdiff=abs(raw-float(s['rho_raw']))
        assert diff<1e-9 and rdiff<1e-12,(m,o,rho,s['rho_adjusted'])
        reports.append(dict(metric=m,outcome=o,status='PASS',rho_adjusted=rho,rho_raw=raw,adjusted_max_difference=diff,raw_max_difference=rdiff))
    output=dict(status='PASS',method='csv manual tied ranks + NumPy pinv (original lstsq) + math.fsum dot product',
        constant_metric_guard='상수 지표는 집단 관련성을 정의할 수 없으므로 원본 극소 잔차 상관을 해석하지 않음',
        comparisons=reports,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'verify_stats_v1.py',HERE/'stats.csv',HERE/'day_features.csv']})
    (HERE/'verification_stats_v1.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
