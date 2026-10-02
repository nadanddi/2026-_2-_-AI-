from support import *
from analyze import pfn
import time
def main():
    while not (OUT/'gpu_done.json').exists() or not (OUT/'plan.json').exists():time.sleep(3)
    jobs=json.loads((OUT/'plan.json').read_text(encoding='utf-8'))['jobs'];core=loadcore();seasoncol=core.FULL.index('day');records=[]
    for job in jobs:
        if job['target']!='EC':continue
        prefix=job['prefix'];z=dict(np.load(OUT/f'{prefix}_cpu.npz'));inp=dict(np.load(OUT/f'{prefix}_input.npz'));b=pd.DataFrame(dict(farm=z['farm'],day=z['day'],hour=z['hour'],sub_ec=z['y']));a=pd.DataFrame(dict(row_id=z['observed_id'],sub_ec=z['observed_y']));a['farm']=a.row_id.str[:3];a['day']=a.row_id.str[4:7].astype(int);bag=pfn(prefix,[1,2,3,4],z['row_id']);split=(z['day']//5)%2
        for seed in ESEEDS:
            p=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi']);target=z['y']-p;x,_,_=state_features(a,b,p);control=np.column_stack([p,inp['q'][:,seasoncol],z['farm']=='F47',z['day']>=179,np.sin(2*np.pi*z['hour']/24),np.cos(2*np.pi*z['hour']/24)])
            for fold in [0,1]:
                tr=split!=fold;va=split==fold;pred=[]
                for design in [control,np.column_stack([control,x])]:
                    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100.));model.fit(design[tr],target[tr]);pred.append(model.predict(design[va]))
                m0=float(np.mean((target[va]-pred[0])**2));m1=float(np.mean((target[va]-pred[1])**2));records.append(dict(prefix=prefix,seed=seed,meta_validation_fold=fold,n_train=int(tr.sum()),n_query=int(va.sum()),control_mse=m0,with_past_mse=m1,delta_mse=m1-m0,delta_pct=100*(np.sqrt(m1/m0)-1)))
    pd.DataFrame(records).to_csv(HERE/'conditional_audit.csv',index=False);savej(HERE/'conditional_audit.json',dict(status='DIAGNOSTIC_ONLY',cells=len(records),improving=sum(r['delta_mse']<0 for r in records),source_hash=sha(__file__),uses_outer_target=False,selection_or_tuning=False,records=records));print(f'CONDITIONAL_AUDIT_DONE {len(records)}',flush=True)
if __name__=='__main__':main()
