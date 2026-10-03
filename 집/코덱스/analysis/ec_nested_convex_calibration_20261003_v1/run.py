from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import numpy as np,pandas as pd
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/ec_nested_convex_calibration_20261003_v1'
def basis(x):return np.column_stack([np.ones(len(x)),x,np.maximum(x-.6,0),np.maximum(x-1,0)])
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    lab,core,wv,folds,outer=S.loadec();rows=[];coeff=[];checks=0
    with threadpool_limits(limits=2):
        for v,k,tm,vm in folds:
            tr=lab[tm];va=lab[vm].reset_index(drop=True)
            z=dict(np.load(S.OUT/f'E_{v}_{k}_cpu.npz'));ids=set(z['row_id']);it=set(z['inner_train_id'])
            assert ids.isdisjoint(it) and ids<=set(tr.row_id) and set(z['outer_train_id']).isdisjoint(va.row_id) and it.isdisjoint(va.row_id)
            b=lab.set_index('row_id').reindex(z['row_id']).reset_index();yday=b.groupby(['farm','day']).sub_ec.mean()
            bag=np.mean([np.load(S.OUT/f'E_{v}_{k}_pfn_{i}.npz')['prediction'] for i in [1,2,3,4]],axis=0);checks+=1
            for seed in [7,101,2024]:
                base=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi'])
                ref=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
                pred=ref.copy();cor=np.zeros(len(va))
                for h in range(24):
                    obs=b.loc[b.hour<=h,['farm','day']].assign(p=base[b.hour<=h]).groupby(['farm','day']).p.mean()
                    x=obs.to_numpy();t=yday.reindex(obs.index).to_numpy()-x;X=basis(x)
                    design=np.vstack([X,np.sqrt(10)*np.eye(4)]);target=np.r_[t,np.zeros(4)]
                    m=lsq_linear(design,target,bounds=([-np.inf,-1,0,0],[np.inf,np.inf,np.inf,np.inf]),tol=1e-12,max_iter=300)
                    assert m.success;beta=m.x
                    grad=design.T@(design@beta-target)
                    lower=np.array([-np.inf,-1,0,0]);active=np.isfinite(lower)&(beta-lower<1e-6)
                    assert np.max(np.abs(grad[~active]))<1e-6
                    assert not active.any() or np.min(grad[active])>-1e-6;checks+=2
                    q=va.loc[va.hour<=h,['farm','day']].assign(p=ref[va.hour<=h]).groupby(['farm','day']).p.mean()
                    use=va.hour==h;idx=pd.MultiIndex.from_frame(va.loc[use,['farm','day']]);xp=q.reindex(idx).to_numpy()
                    cor[use]=.2*np.clip(basis(xp)@beta,-.3,.3)
                    coeff.append(dict(validator=v,fold=k,seed=seed,hour=h,n_days=len(x),beta=beta.tolist(),optimality=m.optimality))
                pred=np.clip(ref+cor,tr.sub_ec.min(),tr.sub_ec.max())
                a=va[['row_id','farm','day','hour']].copy();a['y']=va.sub_ec;a['baseline']=ref;a['candidate']=pred;a['correction']=cor;a['validator']=v;a['fold']=k;a['seed']=seed;a['clip_lo']=tr.sub_ec.min();a['clip_hi']=tr.sub_ec.max();rows.append(a)
            print(v,k,'done',flush=True)
    o=pd.concat(rows,ignore_index=True);o.to_csv(OUT/'oof.csv',index=False)
    (H/'fit_audit_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,coefficients=coeff),ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
