"""H6 cross-fold OOF meta-screen; potentially optimistic, never adoption proof."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'analysis/local'
SEARCH=LOCAL/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=LOCAL/'ec_locked_confirmation/20260928_044934'
FIRST=LOCAL/'tabpfn_cpu_bag/20260927_182052'
EXT=LOCAL/'tabpfn_cpu_bag_extension/20260928_013953'
DATA=ROOT/'온라인대회자료/정형데이터/참가자_배포'
FOLDS=(0,2,4,6,8,9)


def rmse(g,c):
    return float(np.sqrt(np.mean((g.sub_ec-g[c])**2)))


def rows():
    frames=[]
    for fold in FOLDS:
        z=pd.read_csv((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
        z['p']=z['blend'] if fold<8 else z['candidate']
        z['fold']=fold
        path=(FIRST if fold in (0,2) else EXT) if fold<8 else CONFIRM
        contexts=[]
        for seed in (1,2,3,4):
            c=pd.read_csv(path/f'fold{fold}_context{seed}.csv')
            assert c.row_id.tolist()==z.row_id.tolist()
            contexts.append(c.member.to_numpy(float))
        cc=np.stack(contexts,axis=1)
        z['t']=cc.mean(axis=1)
        z['s']=cc.std(axis=1)
        frames.append(z[['row_id','farm','day','fold','sub_ec','p','t','s']])
    x=pd.concat(frames,ignore_index=True)
    assert len(x)==5616 and x.row_id.is_unique
    raw=pd.read_csv(DATA/'train_X.csv',usecols=['row_id','in_temp','act_circfan','act_vent'])
    x=x.merge(raw,on='row_id',validate='one_to_one')
    assert np.isfinite(x[['sub_ec','p','t','s']].to_numpy(float)).all()
    d=x['t']-x['p']
    design=np.column_stack([d,
        d*x.in_temp.ge(10),
        d*(x.act_circfan.lt(10)&x.act_vent.eq(0)),
        d*x.p.ge(.6),d*x.s])
    return x,design.astype(float)


def fit_predict(train,validate,a,b):
    w=np.full(len(a),1/24)
    scale=np.sqrt(np.average(a*a,axis=0,weights=w))
    scale=np.where(scale>1e-12,scale,1)
    a=a/scale;b=b/scale
    target=(train.sub_ec-train.p).to_numpy(float)
    coef=np.linalg.solve(a.T@(w[:,None]*a)+50*np.eye(a.shape[1]),
                         a.T@(w*target))
    return np.clip(b@coef,-.25,.25),coef.tolist(),scale.tolist()


def main():
    x,design=rows()
    pred=np.empty(len(x));info=[]
    for fold in FOLDS:
        trm=x.fold.ne(fold).to_numpy()
        vam=~trm
        tr=x[trm];va=x[vam]
        correction,coef,scale=fit_predict(tr,va,design[trm],design[vam])
        pred[vam]=va.p.to_numpy(float)+correction
        info.append(dict(fold=fold,train_rows=len(tr),validation_rows=len(va),
                         coefficients=coef,scales=scale,
                         mean_absolute_correction=float(np.mean(np.abs(correction)))))
    x['h6']=pred
    assert np.isfinite(x.h6.to_numpy(float)).all()
    metrics={name:dict(rows=len(g),baseline=rmse(g,'p'),candidate=rmse(g,'h6'),
                       relative_change=rmse(g,'h6')/rmse(g,'p')-1)
             for name,g in [('all',x),
                            *[(f'fold{i}',x[x.fold.eq(i)]) for i in FOLDS],
                            *[(f,x[x.farm.eq(f)]) for f in ('F13','F47')]]}
    x['delta_sq']=(x.sub_ec-x.h6)**2-(x.sub_ec-x.p)**2
    day=x.groupby(['farm','day']).delta_sq.mean().to_numpy()
    assert len(day)==234
    rng=np.random.default_rng(2908)
    sample=np.empty(20000)
    for i in range(len(sample)):
        sample[i]=day[rng.integers(len(day),size=len(day))].mean()
    ci=np.quantile(sample,[.025,.975]).tolist()
    passed=all(metrics[f'fold{i}']['relative_change']<0 for i in FOLDS)
    passed=passed and all(metrics[f]['relative_change']<0 for f in ('F13','F47')) and ci[1]<0
    result=dict(info=info,metrics=metrics,day_mse_delta_95ci=ci,
                day_bootstrap_p_worse=float(np.mean(sample>=0)),
                screen_pass=bool(passed),screen_may_have_cross_fold_base_leakage=True,
                test_input_read=False,final_lock_scored=False,submission_created=False)
    out=LOCAL/'ec_member_reliability'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
