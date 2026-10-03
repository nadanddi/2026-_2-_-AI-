"""Post-hoc diagnostic comparator, no fitting or candidate selection."""
from run import *
import math
from sklearn.metrics import roc_auc_score

def main():
    rows=[];checks=[]
    for target,threshold in [('EC',.1),('TEMP',.5)]:
        d=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in OUT.glob(f'RISK_{target}_DIAG10_*.csv')])
        errors=d.assign(e=(d.baseline-d.y)**2).groupby(['farm','day']).e.mean()**.5
        for h in (0,6,12,23):
            q=d[d.hour==h];y=np.array([errors[(f,day)]>threshold for f,day in zip(q.farm,q.day)])
            for name,p in [('risk',q.risk.to_numpy()),('baseline_ascending',q.baseline.to_numpy()),('baseline_descending',-q.baseline.to_numpy())]:
                pos=p[y];neg=p[~y];a=math.fsum(1. if u>v else .5 if u==v else 0. for u in pos for v in neg)/(len(pos)*len(neg))
                assert abs(a-roc_auc_score(y,p))<1e-12
                checks.append(dict(target=target,hour=h,signal=name,passed=True))
                rows.append(dict(target=target,hour=h,signal=name,days=len(y),auc=a))
            brier=math.fsum((float(p)-int(t))**2 for p,t in zip(q.risk,y))/len(y)
            checks.append(dict(target=target,hour=h,passed=abs(brier-float(np.mean((q.risk.to_numpy()-y.astype(int))**2)))<1e-12))
    pd.DataFrame(rows).to_csv(HERE/'risk_simple_comparator_v2.csv',index=False)
    jsonout(HERE/'risk_comparator_verification_v1.json',dict(status='PASS',checks=checks,posthoc=True,selection='both ascending and descending preserved; no model change',scope='DIAG10 only, seed7, fixed context'))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
