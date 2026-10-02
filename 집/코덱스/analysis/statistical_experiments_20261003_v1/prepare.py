from support import *
import time, warnings
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings('ignore',category=UserWarning)

def main():
    start=time.time();jobs=[];audits=[]
    lab,ct,phc,w,folds,_=loadtemp()
    for name,k,fd in folds:
        tm,vm=common.split_mask(lab,fd);tr,va=lab[tm].copy(),lab[vm].copy();im,iv=inner(tr);a,b=tr[im].copy(),tr[iv].copy();iw=w[tm][im]
        prefix=f'T_{name}_{k}';gp=OUT/f'{prefix}_input.npz';cp=OUT/f'{prefix}_cpu.npz'
        if not gp.exists():np.savez(gp,x=a[FEATURE_COLUMNS].to_numpy(np.float32),y=a.sub_temp.to_numpy(),q=b[FEATURE_COLUMNS].to_numpy(np.float32),train_id=a.row_id.to_numpy(dtype=str),query_id=b.row_id.to_numpy(dtype=str),w=iw)
        if not cp.exists():
            payload=dict(row_id=b.row_id.to_numpy(dtype=str),y=b.sub_temp.to_numpy(),w=w[tm][iv],z=conditions(b),gate=gate(b),farm=b.farm.to_numpy(dtype=str),day=b.day.to_numpy(),outer_train_id=tr.row_id.to_numpy(dtype=str),inner_train_id=a.row_id.to_numpy(dtype=str))
            for seed,cs in [(7,726),(101,727)]:
                cold_v5.SEED=seed
                with threadpool_limits(limits=2):
                    mem=temp_members(a,b,ct,phc,iw);payload[f'base_{seed}']=.65*mem['res']+.25*mem['ridge']+.10*mem['nys'];payload[f'codex_{seed}']=TM.codex_fit_predict(a,b,iw,cs)
            np.savez(cp,**payload)
        audits.append(dict(target='TEMP',validator=name,fold=k,outer=gapstats(tr,va),inner=gapstats(a,b)))
        jobs.append(dict(prefix=prefix,target='TEMP',seeds=PSEEDS))
        print(f'{prefix} CPU/inner split ready {len(a)//24}/{len(b)//24} days {time.time()-start:.0f}s',flush=True)
    lab,core,wv,folds,outer=loadec();full=['season' if c=='day' else c for c in core.FULL];base=['season' if c=='day' else c for c in core.BASE]
    for name,k,tm,vm in folds:
        tr,va=lab[tm].copy(),lab[vm].copy();im,iv=inner(tr);a,b=seasonal(tr[im],tr[iv],wv);prefix=f'E_{name}_{k}';gp=OUT/f'{prefix}_input.npz';cp=OUT/f'{prefix}_cpu.npz'
        if not gp.exists():np.savez(gp,x=a[full].to_numpy(np.float32),y=a.sub_ec.to_numpy(),q=b[full].to_numpy(np.float32),train_id=a.row_id.to_numpy(dtype=str),query_id=b.row_id.to_numpy(dtype=str))
        if not cp.exists():
            payload=dict(row_id=b.row_id.to_numpy(dtype=str),y=b.sub_ec.to_numpy(),farm=b.farm.to_numpy(dtype=str),day=b.day.to_numpy(),hour=b.hour.to_numpy(),outer_train_id=tr.row_id.to_numpy(dtype=str),inner_train_id=a.row_id.to_numpy(dtype=str),observed_id=a.row_id.to_numpy(dtype=str),observed_y=a.sub_ec.to_numpy(),lo=float(a.sub_ec.min()),hi=float(a.sub_ec.max()))
            for seed in ESEEDS:
                models=[(core.et(seed),full),(core.lg(seed,'tweedie'),base),(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed)),base)]
                pred=[]
                for model,cols in models:
                    with threadpool_limits(limits=2):model.fit(a[cols],a.sub_ec.to_numpy());pred.append(model.predict(b[cols]))
                payload[f'r3_{seed}']=.6*pred[0]+.3*pred[1]+.1*pred[2]
            np.savez(cp,**payload)
        audits.append(dict(target='EC',validator=name,fold=k,outer=gapstats(tr,va),inner=gapstats(a,b)))
        jobs.append(dict(prefix=prefix,target='EC',seeds=[1,2,3,4]))
        print(f'{prefix} CPU/inner split ready {len(a)//24}/{len(b)//24} days {time.time()-start:.0f}s',flush=True)
    savej(OUT/'plan.json',dict(jobs=jobs,audits=audits,elapsed=time.time()-start,source_hashes={p.name:sha(p) for p in HERE.glob('*.py')}))
    print('CPU_DONE',flush=True)
if __name__=='__main__':main()
