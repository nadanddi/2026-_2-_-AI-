from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
lab,core,wv,folds,outer=S.loadec();rows=[];checks=0
for v,k,tm,vm in folds:
 tr=lab[tm];assert (tr.groupby(['farm','day']).size()==24).all()
 y=tr.sub_ec.to_numpy();d=tr.groupby(['farm','day']).sub_ec.transform('mean').to_numpy();h=y-d
 var=lambda x:float(np.mean((x-np.mean(x))**2))
 vd,vh,vy=var(d),var(h),var(y)
 cov=float(np.mean((d-d.mean())*(h-h.mean())))
 joint=float(np.mean((np.column_stack([y,d])-np.array([y.mean(),d.mean()]))**2))
 assert abs(cov)<1e-12 and abs(vy-vd-vh)<1e-12 and abs(joint-vd-.5*vh)<1e-12;checks+=3
 for arr,expected in [(d,vd),(h,vh),(y,vy)]:
  mu=math.fsum(float(x) for x in arr)/len(arr)
  independent=math.fsum((float(x)-mu)**2 for x in arr)/len(arr)
  assert abs(independent-expected)<1e-12;checks+=1
 rows.append(dict(validator=v,fold=k,train_days=len(tr)//24,current_var=vy,day_mean_var=vd,within_var=vh,joint_root_loss=joint,covariance=cov,current_day_fraction=vd/vy,joint_day_fraction=vd/joint))
R=pd.DataFrame(rows);R.to_csv(H/'target_loss_check_v1.csv',index=False)
(H/'target_loss_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,folds=len(rows),scope='root variance identity on complete training days; not a performance check',threat='After a tree split selects only part of a day, within-day covariance may be nonzero; identity does not generally hold for every node',median_day_fraction=float(R.current_day_fraction.median()),median_joint_day_fraction=float(R.joint_day_fraction.median())),indent=2),encoding='utf-8')
print(R.to_string(index=False));print('CHECKS',checks)
