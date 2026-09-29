"""Fixed v1 comparison versus causal two-stream features. Saves separate proposal."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.analysis-tools/python'))
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from train_initial_model import FARMS,FOLDS,CHECK,PARAMS,signatures,score
from profile_tabular import load,DATA,TARGETS
from hypothesis_features import build

OUT=ROOT/'analysis/local/hypothesis_model_v2'


def predict(x,test):
    joined=pd.concat([x,test],ignore_index=True)
    assert joined.row_id.is_unique
    features,_=build(joined)
    sample=pd.read_csv(DATA/'sample_submission.csv')
    result=sample[['row_id']].copy().set_index('row_id')
    for target in TARGETS:
        result[target]=np.nan
        for farm in FARMS:
            model=joblib.load(OUT/f'{farm}_{target}.joblib')
            ids=test.loc[test.farm.eq(farm),'row_id']
            result.loc[ids,target]=model['model'].predict(features.loc[ids,model['columns']])
    result=result.reset_index()
    assert result.row_id.equals(sample.row_id) and len(result)==1440
    assert result.columns.tolist()==['row_id']+TARGETS
    assert result.row_id.is_unique and np.isfinite(result[TARGETS]).all().all()
    result.to_csv(OUT/'submission.csv',index=False,encoding='utf-8',float_format='%.10f')
    print('Saved hypothesis submission.csv: 1440 rows.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predict-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    x=load('train_X.csv');x=x[x.farm.isin(FARMS)].copy();test=load('test_X.csv')
    if args.predict_only:predict(x,test);return
    y=load('train_y.csv').set_index('row_id')
    features,columns=build(x);meta=x.set_index('row_id').loc[features.index];labels=y.reindex(features.index)
    rows=[];predictions=[];manifest=[]
    cases=[(a,b,'development','blocked') for a,b in FOLDS]+[(*CHECK,'confirmation',mode) for mode in ['blocked','past_only']]
    for start,end,phase,mode in cases:
        for farm in FARMS:
            local=meta.farm.eq(farm);sig=signatures(meta[local].reset_index())
            valid=local&meta.day.ge(start)&meta.day.lt(end)
            reserved=local&meta.day.ge(CHECK[0])&meta.day.lt(CHECK[1])
            excluded=valid|reserved
            matched=sig.index[sig.isin(sig.reindex(meta.loc[excluded,'day'].unique()).dropna())]
            before=local&~excluded
            if mode=='past_only':before&=meta.day.lt(start)
            tr=before&~meta.day.isin(matched)
            assert not (tr&valid).any() and not(tr&reserved).any()
            assert len(set(meta.loc[tr,'day'].map(sig))&set(meta.loc[valid,'day'].map(sig)))==0
            assert valid.sum()==(end-start)*24
            manifest.append(dict(phase=phase,mode=mode,farm=farm,start=start,end=end,n_train=int(tr.sum()),n_valid=int(valid.sum()),purged=int(before.sum()-tr.sum())))
            for target in TARGETS:
                for variant,names in columns[target].items():
                    model=HistGradientBoostingRegressor(**PARAMS)
                    model.fit(features.loc[tr,names],labels.loc[tr,target])
                    p=model.predict(features.loc[valid,names])
                    rows.append(dict(phase=phase,mode=mode,farm=farm,start=start,target=target,variant=variant,**score(labels.loc[valid,target],p)))
                    predictions.extend(dict(phase=phase,mode=mode,row_id=rid,target=target,variant=variant,actual=float(v),prediction=float(w)) for rid,v,w in zip(features.index[valid],labels.loc[valid,target],p))
        pd.DataFrame(rows).to_csv(OUT/'validation.csv',index=False)
        print(f'Completed {phase} {mode} {start}:{end}',flush=True)
    pd.DataFrame(manifest).to_csv(OUT/'fold_manifest.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'validation_predictions.csv',index=False)
    df=pd.DataFrame(rows);df['sse']=df.rmse**2*df.n
    summary=df.groupby(['phase','mode','target','variant']).agg(sse=('sse','sum'),n=('n','sum'))
    summary['rmse']=np.sqrt(summary.sse/summary.n)
    summary.drop(columns='sse').to_csv(OUT/'validation_summary.csv')
    # Ensure baseline and splits exactly reproduce the existing v1 experiment.
    previous=pd.read_csv(ROOT/'analysis/local/initial_model_v1/validation.csv')
    base=df[df.variant.eq('baseline')].copy();base['phase']=base.phase.replace({'development':'selection'})
    previous=previous[((previous.target=='sub_temp')&(previous.variant=='compact'))|((previous.target=='sub_ec')&(previous.variant=='raw'))]
    audit=base.merge(previous,on=['phase','mode','farm','start','target'],suffixes=('_new','_v1'),validate='one_to_one')
    assert len(audit)==len(base) and np.allclose(audit.rmse_new,audit.rmse_v1,atol=1e-10,rtol=0)
    # Always save the requested hypothesis variant, not a post-hoc winner blend.
    final,_=build(pd.concat([x,test],ignore_index=True))
    for farm in FARMS:
        ids=x.loc[x.farm.eq(farm),'row_id']
        for target in TARGETS:
            names=columns[target]['hypothesis']
            model=HistGradientBoostingRegressor(**PARAMS)
            model.fit(final.loc[ids,names],y.loc[ids,target])
            joblib.dump(dict(model=model,columns=names,farm=farm,target=target,params=PARAMS),OUT/f'{farm}_{target}.joblib')
    predict(x,test)
    report=dict(version='hypothesis_model_v2',params=PARAMS,features=columns,
        feature_counts={t:{v:len(c) for v,c in variants.items()} for t,variants in columns.items()},
        baseline_reproduced=True,folds=cases,
        caveat='Previously examined periods, not independent holdout. Causal day-2 input adjacency is a hypothesis, not physical house or calendar recovery. No target lags or cross-farm inputs.',
        summary=summary.drop(columns='sse').reset_index().to_dict('records'),
        source_hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['train_X.csv','train_y.csv','test_X.csv']},
        code_hashes={name:hashlib.sha256((ROOT/'analysis'/name).read_bytes()).hexdigest() for name in ['hypothesis_features.py','initial_features.py','train_hypothesis_model.py','train_initial_model.py']})
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.drop(columns='sse').to_string());print(report['feature_counts'])


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
