from support import *
from analyze import pfn
import time,math
def main():
    while not (HERE/'conditional_audit.json').exists():time.sleep(3)
    result=json.loads((HERE/'conditional_audit.json').read_text(encoding='utf-8'));saved={(r['prefix'],r['seed'],r['meta_validation_fold']):r for r in result['records']};core=loadcore();seasoncol=core.FULL.index('day');gaps=[];records=[]
    for prefix in sorted({r['prefix'] for r in result['records']}):
        z=dict(np.load(OUT/f'{prefix}_cpu.npz'));inp=dict(np.load(OUT/f'{prefix}_input.npz'));b=pd.DataFrame(dict(farm=z['farm'],day=z['day'],hour=z['hour'],sub_ec=z['y']));a=pd.DataFrame(dict(row_id=z['observed_id'],sub_ec=z['observed_y']));a['farm']=a.row_id.str[:3];a['day']=a.row_id.str[4:7].astype(int);bag=pfn(prefix,[1,2,3,4],z['row_id']);split=(z['day']//5)%2
        for seed in ESEEDS:
            p=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi']);target=z['y']-p;x,_,_=state_features(a,b,p);control=np.column_stack([p,inp['q'][:,seasoncol],z['farm']=='F47',z['day']>=179,np.sin(2*np.pi*z['hour']/24),np.cos(2*np.pi*z['hour']/24)])
            for fold in [0,1]:
                tr=split!=fold;va=split==fold;mses=[]
                for design in [control,np.column_stack([control,x])]:
                    # All these controls are finite; independent manual scaler + normal equations.
                    assert np.isfinite(design).all();mu=design[tr].mean(0);sd=design[tr].std(0);sd[sd==0]=1;xt=(design[tr]-mu)/sd;xq=(design[va]-mu)/sd;ym=target[tr].mean();beta=np.linalg.solve(xt.T@xt+100*np.eye(xt.shape[1]),xt.T@(target[tr]-ym));pred=xq@beta+ym;mses.append(math.fsum((float(t)-float(q))**2 for t,q in zip(target[va],pred))/int(va.sum()))
                old=saved[(prefix,seed,fold)];difference=max(abs(mses[0]-old['control_mse']),abs(mses[1]-old['with_past_mse']));assert difference<1e-10;gaps.append(difference);records.append(dict(prefix=prefix,seed=seed,fold=fold,delta_mse=mses[1]-mses[0]))
    assert len(records)==132;savej(HERE/'conditional_verification.json',dict(status='PASS',cells=len(records),improving=sum(r['delta_mse']<0 for r in records),max_mse_difference=max(gaps),method='manual train-only scaler + normal equations + math.fsum',records=records));print('CONDITIONAL_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
