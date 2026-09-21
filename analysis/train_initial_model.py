"""Initial model v1: purged block validation, four saved estimators, local CSV.

Run with --predict-only to reproduce submission from saved model files.
Validation uses train_X only. Final fit uses train_X plus permitted test_X history.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.analysis-tools/python'))
import argparse
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from profile_tabular import load, DATA, TARGETS
from initial_features import build

OUT=ROOT/'analysis/local/initial_model_v1'
FARMS=['F13','F47']
FOLDS=[(60,70),(120,130),(150,160)]
CHECK=(170,180)
PARAMS=dict(max_iter=160,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,
            l2_regularization=10.,early_stopping=False,random_state=20260919)


def signatures(x):
    p=x.pivot(index='day',columns='hour',values=['out_temp','out_hum','out_rad','out_wspd'])
    assert p.shape[1]==96 and p.notna().all().all()
    h=pd.util.hash_pandas_object(p,index=False)
    for ids in h.groupby(h).groups.values():
        a=p.loc[ids].to_numpy();assert np.array_equal(a,np.repeat(a[:1],len(a),axis=0))
    return h


def score(y,p):
    return {'n':len(y),'rmse':float(np.sqrt(np.mean((np.asarray(y)-p)**2))),
            'mae':float(np.mean(np.abs(np.asarray(y)-p)))}


def predict_saved(x,test,report):
    combined=pd.concat([x,test],ignore_index=True)
    assert combined.row_id.is_unique
    features,_=build(combined)
    sample=pd.read_csv(DATA/'sample_submission.csv')
    assert sample.row_id.is_unique and set(sample.row_id)==set(test.row_id)
    result=sample[['row_id']].copy().set_index('row_id')
    for target in TARGETS:
        result[target]=np.nan
        for farm in FARMS:
            pack=joblib.load(OUT/f'{farm}_{target}.joblib')
            ids=test.loc[test.farm.eq(farm),'row_id']
            result.loc[ids,target]=pack['model'].predict(features.loc[ids,pack['columns']])
    result=result.reset_index()
    assert result.columns.tolist()==['row_id']+TARGETS
    assert len(result)==1440 and result.row_id.equals(sample.row_id)
    assert np.isfinite(result[TARGETS].to_numpy()).all()
    result.to_csv(OUT/'submission.csv',index=False,encoding='utf-8',float_format='%.10f')
    reread=pd.read_csv(OUT/'submission.csv')
    assert reread.row_id.equals(sample.row_id) and np.isfinite(reread[TARGETS]).all().all()
    report['predictions']={t:{'min':float(result[t].min()),'max':float(result[t].max()),'mean':float(result[t].mean())} for t in TARGETS}
    print('Saved 1440-row submission.csv; no upload performed.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predict-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    x=load('train_X.csv');x=x[x.farm.isin(FARMS)].copy()
    test=load('test_X.csv');y=load('train_y.csv').set_index('row_id')
    if args.predict_only:
        report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
        predict_saved(x,test,report);return
    features,groups=build(x)
    meta=x.set_index('row_id').loc[features.index]
    labels=y.reindex(features.index)
    assert labels[TARGETS].notna().all().all()
    results=[];predictions=[];manifest=[]
    def evaluate(start,end,phase,mode,selected=None):
        for farm in FARMS:
            local=meta.farm.eq(farm)
            sig=signatures(meta[local].reset_index())
            valid=local&meta.day.ge(start)&meta.day.lt(end)
            # Confirmation labels are withheld from every selection-stage fit.
            reserved=local&meta.day.ge(CHECK[0])&meta.day.lt(CHECK[1])
            excluded=valid|reserved
            forbidden=sig.reindex(meta.loc[excluded,'day'].unique()).dropna()
            matched=sig.index[sig.isin(forbidden)]
            before=local&~excluded
            if mode=='past_only':before&=meta.day.lt(start)
            tr=before&~meta.day.isin(matched)
            assert not (tr&valid).any() and not (tr&reserved).any()
            assert len(set(meta.loc[tr,'day'].map(sig))&set(meta.loc[valid,'day'].map(sig)))==0
            assert valid.sum()==(end-start)*24, (farm,start,end,int(valid.sum()))
            assert tr.sum()>100
            manifest.append(dict(phase=phase,mode=mode,farm=farm,start=start,end=end,n_train=int(tr.sum()),n_valid=int(valid.sum()),purged=int(before.sum()-tr.sum())))
            for target in TARGETS:
                sets=['raw','compact'] if selected is None else [selected[target]]
                mean=np.repeat(labels.loc[tr,target].mean(),valid.sum())
                entries=[('train_mean',mean)]
                for name in sets:
                    columns=groups['raw'] if name=='raw' else groups[target]
                    model=HistGradientBoostingRegressor(**PARAMS)
                    model.fit(features.loc[tr,columns],labels.loc[tr,target])
                    entries.append((name,model.predict(features.loc[valid,columns])))
                for name,pred in entries:
                    results.append(dict(phase=phase,mode=mode,farm=farm,start=start,target=target,variant=name,**score(labels.loc[valid,target],pred)))
                    predictions.extend(dict(phase=phase,mode=mode,row_id=rid,target=target,variant=name,actual=float(actual),prediction=float(p)) for rid,actual,p in zip(features.index[valid],labels.loc[valid,target],pred))
        pd.DataFrame(results).to_csv(OUT/'validation.csv',index=False)
        print(f'Completed {phase} {mode} days {start}:{end}',flush=True)
    for a,b in FOLDS:evaluate(a,b,'selection','blocked')
    df=pd.DataFrame(results);df['sse']=df.rmse**2*df.n
    aggregate=df.groupby(['target','variant']).agg(sse=('sse','sum'),n=('n','sum'))
    aggregate['rmse']=np.sqrt(aggregate.sse/aggregate.n)
    chosen={t:aggregate.loc[t].loc[['raw','compact'],'rmse'].idxmin() for t in TARGETS}
    print('Selected feature sets:',chosen,flush=True)
    evaluate(*CHECK,'confirmation','blocked',selected=chosen)
    evaluate(*CHECK,'confirmation','past_only',selected=chosen)
    pd.DataFrame(predictions).to_csv(OUT/'validation_predictions.csv',index=False)
    pd.DataFrame(manifest).to_csv(OUT/'fold_manifest.csv',index=False)
    all_scores=pd.DataFrame(results);all_scores['sse']=all_scores.rmse**2*all_scores.n
    summary=all_scores.groupby(['phase','mode','target','variant']).agg(sse=('sse','sum'),n=('n','sum'))
    summary['rmse']=np.sqrt(summary.sse/summary.n)
    summary.drop(columns='sse').to_csv(OUT/'validation_summary.csv')
    # train/test input history is permitted; all labels come strictly from train_y.
    combined=pd.concat([x,test],ignore_index=True)
    final_features,_=build(combined)
    for farm in FARMS:
        ids=x.loc[x.farm.eq(farm),'row_id']
        for target in TARGETS:
            columns=groups['raw'] if chosen[target]=='raw' else groups[target]
            model=HistGradientBoostingRegressor(**PARAMS)
            model.fit(final_features.loc[ids,columns],y.loc[ids,target])
            joblib.dump(dict(model=model,columns=columns,farm=farm,target=target,variant=chosen[target],params=PARAMS),OUT/f'{farm}_{target}.joblib')
    report=dict(version='initial_model_v1',params=PARAMS,selected=chosen,feature_counts={k:len(v) for k,v in groups.items()},features=groups,
        selection_folds=FOLDS,confirmation_fold=CHECK,
        limitation='Historical EDA already examined these periods; confirmation is withheld from this run selection, not a pristine unseen benchmark. Validation history uses train_X only; final history includes available test_X.',
        versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__),
        hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['train_X.csv','train_y.csv','test_X.csv','sample_submission.csv']},
        code_hashes={name:hashlib.sha256((ROOT/'analysis'/name).read_bytes()).hexdigest() for name in ['initial_features.py','train_initial_model.py','causal_features.py']},
        validation_summary=summary.drop(columns='sse').reset_index().to_dict('records'))
    predict_saved(x,test,report)
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.drop(columns='sse').to_string(),flush=True)
    print('Feature counts:',report['feature_counts'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
