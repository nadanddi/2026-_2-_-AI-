from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
import run as R
import pandas as pd,numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
lab,core,_,_,_=R.S.loadec();z=dict(np.load(R.S.OUT/'E_DIAG10_0_cpu.npz'))
b=lab.set_index('row_id').reindex(z['row_id']).reset_index();bag=np.mean([np.load(R.S.OUT/f'E_DIAG10_0_pfn_{i}.npz')['prediction'] for i in [1,2,3,4]],axis=0)
audit=json.loads((H/'fit_audit_v1.json').read_text(encoding='utf-8'));o=pd.read_csv(R.OUT/'oof.csv',float_precision='round_trip');maxobj=0;maxpred=0;checks=0
with threadpool_limits(limits=2):
    for seed in [7,101,2024]:
        ref=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi'])
        yd=b.groupby(['farm','day']).sub_ec.mean()
        for h in [0,23]:
            p=b.loc[b.hour<=h,['farm','day']].assign(p=ref[b.hour<=h]).groupby(['farm','day']).p.mean()
            x=p.to_numpy();y=yd.reindex(p.index).to_numpy()-x;X=R.basis(x)
            fun=lambda beta:float(np.sum((X@beta-y)**2)+10*np.dot(beta,beta))
            jac=lambda beta:2*(X.T@(X@beta-y)+10*beta)
            opt=minimize(fun,np.zeros(4),jac=jac,method='SLSQP',bounds=[(None,None),(-1,None),(0,None),(0,None)],options=dict(ftol=1e-12,maxiter=1000))
            beta=np.array(next(c['beta'] for c in audit['coefficients'] if c['validator']=='DIAG10' and c['fold']==0 and c['seed']==seed and c['hour']==h))
            assert opt.success;dif=abs(fun(opt.x)-fun(beta));maxobj=max(maxobj,dif);assert dif<1e-8
            dif=float(np.max(np.abs(X@opt.x-X@beta)));maxpred=max(maxpred,dif);assert dif<1e-6;checks+=2
            g=o[(o.validator=='DIAG10')&(o.fold==0)&(o.seed==seed)];q=g[g.hour<=h].groupby(['farm','day']).baseline.mean()
            for row in g[g.hour==h].itertuples():
                xp=q.loc[(row.farm,row.day)];pred=.2*max(-.3,min(.3,float(R.basis(np.array([xp]))@beta)));assert abs(pred-row.correction)<1e-12;checks+=1
(H/'fit_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,independent_optimizer='SLSQP versus constrained least squares',objective_maxdiff=maxobj,prediction_maxdiff=maxpred),indent=2),encoding='utf-8');print('PASS',checks,maxobj,maxpred)
