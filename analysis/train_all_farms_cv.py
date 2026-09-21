"""All-farm supervised learning, purged 10-day-block 5-fold CV and saved models."""
from train_initial_model import *

DEST=ROOT/'analysis/local/all_farms_cv_v3'
RAW19=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','in_rad',
       'act_vent','act_side','act_shade','act_thermal','act_valve','act_heating',
       'act_circfan','act_co2','act_fog','act_cool','act_pump']

def features(x):
    f,g=build(x);m=x.set_index('row_id').loc[f.index]
    for c in RAW19:
        f[c]=m[c]
        f[c+'_absent']=m[c].isna().astype(float)
    f['farm_code']=m.farm.str[1:].astype(int)
    common=[c for c in g['sub_temp'] if c.startswith(('in_temp','in_hum','in_co2'))]
    common+=['day','hour_sin','hour_cos','midnight','farm_code']
    full=list(dict.fromkeys(g['sub_temp']+RAW19+[c+'_absent' for c in RAW19]+['farm_code']))
    ec=list(dict.fromkeys(g['raw']+RAW19+[c+'_absent' for c in RAW19]+['farm_code']))
    return f,dict(common=common,full=full,ec=ec,separate_temp=g['sub_temp'],separate_ec=g['raw'])

def folds(x):
    m=x.set_index('row_id');start=m.groupby('farm').day.transform('min')
    assignment=((m.day-start)//10%5).astype(int)
    # Full-day external signatures are used only for split purging, never prediction.
    weather=pd.Series(index=m.index,dtype='object')
    for farm in FARMS:
        local=m.farm.eq(farm);s=signatures(m[local].reset_index())
        weather.loc[local]=m.loc[local,'day'].map(s).astype(str)
    # Remove exact input duplicates across folds, including known F10/F11 duplicates.
    hashes=pd.util.hash_pandas_object(m[RAW19],index=False)
    result=[]
    for k in range(5):
        valid=assignment.eq(k);train=~valid
        for farm in m.farm.unique():
            vd=m.loc[valid&m.farm.eq(farm),'day'].unique()
            near=set(vd)|set(vd-1)|set(vd+1)
            train &= ~(m.farm.eq(farm)&m.day.isin(near))
        forbidden=set(weather[valid].dropna())
        train &= ~weather.isin(forbidden)
        train &= ~hashes.isin(hashes[valid])
        assert not (train&valid).any()
        assert not set(weather[train].dropna())&forbidden
        assert not set(hashes[train])&set(hashes[valid])
        result.append((train,valid))
    assert np.all(np.sum([v.to_numpy() for _,v in result],axis=0)==1)
    return assignment,result

def estimator(target,pooled):
    params=PARAMS.copy()
    if target=='sub_temp':params.update(max_iter=300,max_leaf_nodes=23,min_samples_leaf=20)
    if pooled:params['categorical_features']=['farm_code']
    return HistGradientBoostingRegressor(**params)

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    x=load('train_X.csv');y=load('train_y.csv').set_index('row_id')
    f,g=features(x);m=x.set_index('row_id').loc[f.index];labels=y.reindex(f.index)
    assignment,splits=folds(x);assignment=assignment.reindex(f.index)
    inventory=m[['farm']].join(labels[TARGETS]).groupby('farm')[TARGETS].count()
    inventory.to_csv(DEST/'label_counts.csv')
    print('Labels:',inventory.sum().to_dict(),'farms:',(inventory>0).sum().to_dict(),flush=True)
    specs={'sub_temp':{'separate':('separate_temp',False,1),'all_common':('common',True,1),
                      'all_full':('full',True,1),'all_target_weight20':('full',True,20)},
           'sub_ec':{'separate':('separate_ec',False,1),'pooled':('ec',True,1)}}
    oof={t:{name:pd.Series(np.nan,index=f.index) for name in sp} for t,sp in specs.items()}
    manifest=[]
    for k,(tr0,va0) in enumerate(splits):
        tr0=tr0.reindex(f.index);va0=va0.reindex(f.index)
        for target,sp in specs.items():
            train=tr0&labels[target].notna();valid=va0&labels[target].notna()
            for name,(group,pooled,weight) in sp.items():
                scope=[None] if pooled else FARMS
                for farm in scope:
                    tr=train.copy();va=valid.copy()
                    if farm is not None:tr &= m.farm.eq(farm);va &= m.farm.eq(farm)
                    model=estimator(target,pooled)
                    w=np.where(m.loc[tr,'farm'].isin(FARMS),weight,1.)
                    cols=[c for c in g[group] if f.loc[tr,c].notna().any()]
                    model.fit(f.loc[tr,cols],labels.loc[tr,target],sample_weight=w)
                    oof[target][name].loc[va]=model.predict(f.loc[va,cols])
                    manifest.append(dict(fold=k,target=target,variant=name,farm=farm or 'pooled',n_train=int(tr.sum()),n_valid=int(va.sum())))
                print(f'fold {k+1}/5 {target} {name} done',flush=True)
        joblib.dump(oof,DEST/'oof_checkpoint.joblib')
    rows=[];predrows=[]
    for target,variants in oof.items():
        targetmask=m.farm.isin(FARMS)&labels[target].notna()
        assert all(p[targetmask].notna().all() for p in variants.values())
        if target=='sub_temp':
            for source in ['all_common','all_full','all_target_weight20']:
                for alpha in [.25,.5,.75]:
                    variants[f'blend_{source}_{alpha}']=(1-alpha)*variants['separate']+alpha*variants[source]
        for name,p in variants.items():
            for k in list(range(5))+['all']:
                for scope in ['targets','all_labeled','F13','F47']:
                    va=p.notna()&labels[target].notna()
                    if k!='all':va &= assignment.eq(k)
                    if scope=='targets':va &= m.farm.isin(FARMS)
                    elif scope in FARMS:va &= m.farm.eq(scope)
                    rows.append(dict(target=target,variant=name,fold=k,scope=scope,**score(labels.loc[va,target],p[va].to_numpy())))
            ids=f.index[targetmask]
            predrows.extend(dict(row_id=rid,fold=int(assignment[rid]),target=target,variant=name,actual=float(labels.at[rid,target]),prediction=float(p[rid])) for rid in ids)
    scores=pd.DataFrame(rows);scores.to_csv(DEST/'scores.csv',index=False)
    pd.DataFrame(predrows).to_csv(DEST/'target_oof_predictions.csv',index=False)
    pd.DataFrame(manifest).to_csv(DEST/'fold_manifest.csv',index=False)
    summary=scores[(scores.fold=='all')&(scores.scope=='targets')]
    chosen={t:summary[summary.target.eq(t)].sort_values('rmse').iloc[0]['variant'] for t in TARGETS}
    print(summary.to_string(index=False),flush=True);print('Chosen:',chosen,flush=True)
    # Final candidates use every available training label, including validation rows.
    test=load('test_X.csv');ff,_=features(pd.concat([x,test],ignore_index=True))
    sample=pd.read_csv(DATA/'sample_submission.csv');result=sample[['row_id']].copy().set_index('row_id')
    packs={}
    for target,name in chosen.items():
        if name.startswith('blend_'):
            source,alpha=name[6:].rsplit('_',1);parts=[('separate',1-float(alpha)),(source,float(alpha))]
        else:parts=[(name,1.)]
        total=pd.Series(0.,index=test.row_id)
        for variant,coef in parts:
            group,pooled,weight=specs[target][variant]
            for farm in ([None] if pooled else FARMS):
                tr=labels[target].notna()
                ids=test.row_id
                if farm is not None:tr &= m.farm.eq(farm);ids=test.loc[test.farm.eq(farm),'row_id']
                model=estimator(target,pooled);train_ids=f.index[tr]
                cols=[c for c in g[group] if ff.loc[train_ids,c].notna().any()]
                model.fit(ff.loc[train_ids,cols],labels.loc[tr,target],sample_weight=np.where(m.loc[tr,'farm'].isin(FARMS),weight,1.))
                p=model.predict(ff.loc[ids,cols])
                total.loc[ids]+=coef*p
                key=f'{target}_{variant}_{farm or "pooled"}'
                packs[key]=dict(model=model,columns=cols,coef=coef,target=target,farm=farm,n_train=len(train_ids))
                print('Final fit',key,len(train_ids),flush=True)
        result[target]=total.reindex(result.index)
    joblib.dump(packs,DEST/'models.joblib')
    # Saved-model round trip and schema checks.
    check=pd.DataFrame(0.,index=result.index,columns=TARGETS)
    for p in joblib.load(DEST/'models.joblib').values():
        ids=test.row_id if p['farm'] is None else test.loc[test.farm.eq(p['farm']),'row_id']
        check.loc[ids,p['target']]+=p['coef']*p['model'].predict(ff.loc[ids,p['columns']])
    np.testing.assert_allclose(check[TARGETS],result[TARGETS],rtol=0,atol=1e-12)
    assert len(result)==1440 and result.index.tolist()==sample.row_id.tolist()
    assert np.isfinite(result[TARGETS]).all().all()
    result.reset_index().to_csv(DEST/'submission.csv',index=False,float_format='%.10f')
    report=dict(chosen=chosen,features=g,specs=specs,summary=summary.to_dict('records'),
        labels=inventory.sum().to_dict(),labeled_farms=(inventory>0).sum().to_dict(),
        split='Per-farm 10-day blocks, round-robin 5 folds; 1-day embargo; cross-target-farm weather and exact input duplicate purge. Each labeled row validates once for pooled models.',
        limitation='CV is used for selection, not independent final assessment. Blocks are relative per farm, not synchronized real dates. EC labels absent in 49 farms; these cannot directly train supervised EC. Missing-label input rows remain available as feature history.',
        hashes={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['train_X.csv','train_y.csv','test_X.csv']},
        code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(sklearn=sklearn.__version__,pandas=pd.__version__,numpy=np.__version__))
    (DEST/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
