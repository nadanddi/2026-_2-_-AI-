"""Fixed v3 folds: LightGBM temperature and ExtraTrees EC comparison."""
from train_all_farms_cv import *
from lightgbm import LGBMRegressor
import lightgbm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

OUT4=ROOT/'analysis/local/lgbm_extratrees_v4'
LGB=dict(n_estimators=300,learning_rate=.05,num_leaves=23,min_child_samples=20,
         reg_lambda=10.,random_state=20260919,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
ET=dict(n_estimators=300,max_features=1.,random_state=20260919,n_jobs=2)

def fit(target,variant,X,y,weight):
    if target=='sub_temp':
        model=LGBMRegressor(**LGB)
        model.fit(X,y,sample_weight=weight,categorical_feature=['farm_code'] if 'farm_code' in X else [])
    else:
        model=make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(**ET,min_samples_leaf=int(variant.split('_')[-1])))
        model.fit(X,y,extratreesregressor__sample_weight=weight)
    return model

def main():
    OUT4.mkdir(exist_ok=True,parents=True)
    x=load('train_X.csv');y=load('train_y.csv').set_index('row_id')
    f,g=features(x);m=x.set_index('row_id').loc[f.index];labels=y.reindex(f.index)
    assignment,splits=folds(x);assignment=assignment.reindex(f.index)
    specs={'sub_temp':{'separate':('separate_temp',False,1),'pooled':('full',True,20)},
           'sub_ec':{'leaf_2':('ec',True,1),'leaf_10':('ec',True,1)}}
    oof={t:{n:pd.Series(np.nan,index=f.index) for n in s} for t,s in specs.items()}
    for k,(tr0,va0) in enumerate(splits):
        for target,sp in specs.items():
            for name,(group,pooled,weight) in sp.items():
                for farm in ([None] if pooled else FARMS):
                    tr=tr0.reindex(f.index)&labels[target].notna()
                    va=va0.reindex(f.index)&labels[target].notna()&m.farm.isin(FARMS)
                    if farm is not None:tr &= m.farm.eq(farm);va &= m.farm.eq(farm)
                    cols=[c for c in g[group] if f.loc[tr,c].notna().any()]
                    model=fit(target,name,f.loc[tr,cols],labels.loc[tr,target],np.where(m.loc[tr,'farm'].isin(FARMS),weight,1.))
                    oof[target][name].loc[va]=model.predict(f.loc[va,cols])
                print(f'fold {k+1}/5 {target} {name} done',flush=True)
    oof['sub_temp']['blend_50']=(oof['sub_temp']['separate']+oof['sub_temp']['pooled'])/2
    rows=[];predictions=[]
    for target,variants in oof.items():
        for name,p in variants.items():
            mask=labels[target].notna()&m.farm.isin(FARMS)
            assert p[mask].notna().all()
            for k in list(range(5))+['all']:
                for scope in ['targets']+FARMS:
                    va=mask.copy()
                    if k!='all':va &= assignment.eq(k)
                    if scope!='targets':va &= m.farm.eq(scope)
                    rows.append(dict(target=target,variant=name,fold=k,scope=scope,**score(labels.loc[va,target],p[va].to_numpy())))
            predictions.extend(dict(row_id=rid,fold=int(assignment[rid]),target=target,variant=name,actual=float(labels.at[rid,target]),prediction=float(p[rid])) for rid in f.index[mask])
    scores=pd.DataFrame(rows);scores.to_csv(OUT4/'scores.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT4/'oof_predictions.csv',index=False)
    summary=scores[(scores.fold=='all')&scores.scope.eq('targets')]
    chosen={t:summary[summary.target.eq(t)].sort_values('rmse').iloc[0]['variant'] for t in TARGETS}
    old=pd.read_csv(DEST/'target_oof_predictions.csv')
    for t in TARGETS:
        ref=old[(old.target==t)&(old.variant==('blend_all_target_weight20_0.5' if t=='sub_temp' else 'pooled'))].set_index('row_id')
        ids=f.index[m.farm.isin(FARMS)&labels[t].notna()]
        np.testing.assert_array_equal(ref.loc[ids,'fold'],assignment.loc[ids])
        np.testing.assert_allclose(ref.loc[ids,'actual'],labels.loc[ids,t],rtol=0,atol=1e-12)
    print(summary.to_string(index=False),flush=True);print('Chosen requested algorithms:',chosen,flush=True)
    test=load('test_X.csv');ff,_=features(pd.concat([x,test],ignore_index=True))
    sample=pd.read_csv(DATA/'sample_submission.csv');result=sample[['row_id']].set_index('row_id');packs={}
    for target,name in chosen.items():
        parts=[('separate',.5),('pooled',.5)] if name=='blend_50' else [(name,1.)]
        total=pd.Series(0.,index=result.index)
        for variant,coef in parts:
            group,pooled,weight=specs[target][variant]
            for farm in ([None] if pooled else FARMS):
                tr=labels[target].notna();ids=test.row_id
                if farm is not None:tr &= m.farm.eq(farm);ids=test.loc[test.farm.eq(farm),'row_id']
                trainids=f.index[tr];cols=[c for c in g[group] if ff.loc[trainids,c].notna().any()]
                model=fit(target,variant,ff.loc[trainids,cols],labels.loc[tr,target],np.where(m.loc[tr,'farm'].isin(FARMS),weight,1.))
                total.loc[ids]+=coef*model.predict(ff.loc[ids,cols])
                packs[f'{target}_{variant}_{farm}']=dict(model=model,columns=cols,target=target,farm=farm,coef=coef,n_train=len(trainids))
                print('Final fit:',target,variant,farm,len(trainids),flush=True)
        result[target]=total
    joblib.dump(packs,OUT4/'models.joblib')
    check=pd.DataFrame(0.,index=result.index,columns=TARGETS)
    for p in joblib.load(OUT4/'models.joblib').values():
        ids=test.row_id if p['farm'] is None else test.loc[test.farm.eq(p['farm']),'row_id']
        check.loc[ids,p['target']]+=p['coef']*p['model'].predict(ff.loc[ids,p['columns']])
    np.testing.assert_allclose(check,result[TARGETS],rtol=0,atol=1e-12)
    assert len(result)==1440 and result.index.tolist()==sample.row_id.tolist()
    assert np.isfinite(result[TARGETS]).all().all()
    result.reset_index().to_csv(OUT4/'submission.csv',index=False,float_format='%.10f')
    (OUT4/'report.json').write_text(json.dumps(dict(chosen=chosen,summary=summary.to_dict('records'),LGB=LGB,ET=ET,
        versions=dict(lightgbm=lightgbm.__version__,sklearn=sklearn.__version__),
        hashes={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['train_X.csv','train_y.csv','test_X.csv']},
        limitation='Same v3 CV used for comparison and selection, not independent evaluation. Requested algorithm candidates saved regardless of whether they outperform v3.'),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
