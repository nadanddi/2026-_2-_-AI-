"""Confirm EC removal candidates and build v5 while preserving v4 temperature."""
from feature_group_ablation import *

OUT5=ROOT/'analysis/local/ec_refined_v5'
CANDIDATES={
    'baseline':set(),
    'no_farm_identity':{'farm_identity'},
    'no_data_quality':{'data_quality'},
    'no_farm_and_quality':{'farm_identity','data_quality'},
}

def main():
    OUT5.mkdir(parents=True,exist_ok=True)
    x=load('train_X.csv');y=load('train_y.csv').set_index('row_id')
    f,g=features(x);m=x.set_index('row_id').loc[f.index];labels=y.reindex(f.index)
    assignment,splits=folds(x);assignment=assignment.reindex(f.index)
    old=pd.read_csv(ROOT/'analysis/local/feature_group_ablation_v4/ablation_predictions.csv')
    oof={}
    base=pd.read_csv(ROOT/'analysis/local/lgbm_extratrees_v4/oof_predictions.csv')
    q=base[(base.target=='sub_ec')&base.variant.eq('leaf_2')].set_index('row_id')
    oof['baseline']=q.prediction.reindex(f.index)
    for name,group in [('no_farm_identity','farm_identity'),('no_data_quality','data_quality')]:
        q=old[(old.target=='sub_ec')&old.group.eq(group)].set_index('row_id')
        oof[name]=q.prediction.reindex(f.index)
    combo=pd.Series(np.nan,index=f.index)
    for k,(tr0,va0) in enumerate(splits):
        tr=tr0.reindex(f.index)&labels.sub_ec.notna()
        va=va0.reindex(f.index)&labels.sub_ec.notna()&m.farm.isin(FARMS)
        cols=[c for c in g['ec'] if group_for(c) not in CANDIDATES['no_farm_and_quality'] and f.loc[tr,c].notna().any()]
        model=fit('sub_ec','leaf_2',f.loc[tr,cols],labels.loc[tr,'sub_ec'],np.ones(tr.sum()))
        combo.loc[va]=model.predict(f.loc[va,cols])
        print(f'fold {k+1}/5 combined removal done',flush=True)
    oof['no_farm_and_quality']=combo
    mask=labels.sub_ec.notna()&m.farm.isin(FARMS);actual=labels.loc[mask,'sub_ec']
    rows=[]
    basepred=oof['baseline'].loc[mask]
    clusters=(m.loc[mask,'farm']+'_'+m.loc[mask,'day'].astype(str)).to_numpy()
    for i,(name,p) in enumerate(oof.items()):
        pred=p.loc[mask];rm=score(actual,pred.to_numpy())['rmse']
        ci=[0.,0.,0.] if name=='baseline' else cluster_bootstrap(actual.to_numpy(),basepred.to_numpy(),pred.to_numpy(),clusters,20261001+i)
        foldscores=[]
        for k in range(5):
            z=assignment.loc[mask].eq(k);foldscores.append(score(actual[z],pred[z].to_numpy())['rmse'])
        rows.append(dict(variant=name,rmse=rm,delta_vs_baseline=rm-score(actual,basepred.to_numpy())['rmse'],ci_low=ci[0],ci_high=ci[2],fold_rmses=';'.join(f'{v:.8f}' for v in foldscores)))
    summary=pd.DataFrame(rows).sort_values('rmse');summary.to_csv(OUT5/'summary.csv',index=False)
    chosen=summary.iloc[0].variant;assert chosen!='baseline'
    pd.DataFrame({name:p.loc[mask] for name,p in oof.items()}).assign(actual=actual).to_csv(OUT5/'oof_predictions.csv')
    print(summary.to_string(index=False),flush=True);print('Chosen:',chosen,flush=True)
    # Final EC model; keep the verified v4 temperature predictions and model packs.
    test=load('test_X.csv');ff,_=features(pd.concat([x,test],ignore_index=True));tr=labels.sub_ec.notna()
    removed=CANDIDATES[chosen]
    cols=[c for c in g['ec'] if group_for(c) not in removed and ff.loc[f.index[tr],c].notna().any()]
    ec_model=fit('sub_ec','leaf_2',ff.loc[f.index[tr],cols],labels.loc[tr,'sub_ec'],np.ones(tr.sum()))
    v4=pd.read_csv(ROOT/'analysis/local/lgbm_extratrees_v4/submission.csv').set_index('row_id')
    ids=test.row_id;ecpred=ec_model.predict(ff.loc[ids,cols]);v4.loc[ids,'sub_ec']=ecpred
    sample=pd.read_csv(DATA/'sample_submission.csv')
    assert v4.index.tolist()==sample.row_id.tolist() and np.isfinite(v4[TARGETS]).all().all()
    v4.reset_index().to_csv(OUT5/'submission.csv',index=False,float_format='%.10f')
    v4packs=joblib.load(ROOT/'analysis/local/lgbm_extratrees_v4/models.joblib')
    packs={k:v for k,v in v4packs.items() if v['target']=='sub_temp'}
    packs['sub_ec_refined_pooled']=dict(model=ec_model,columns=cols,target='sub_ec',farm=None,coef=1.,n_train=int(tr.sum()),removed_groups=sorted(removed))
    joblib.dump(packs,OUT5/'models.joblib')
    loaded=joblib.load(OUT5/'models.joblib')['sub_ec_refined_pooled']
    np.testing.assert_allclose(loaded['model'].predict(ff.loc[ids,loaded['columns']]),ecpred,rtol=0,atol=1e-12)
    (OUT5/'report.json').write_text(json.dumps(dict(chosen=chosen,summary=summary.to_dict('records'),removed_groups=sorted(removed),
        temperature='Unchanged v4 LightGBM predictions and model packs.',bootstrap_repetitions=BOOTSTRAPS,
        limitation='Candidate selection and evaluation reuse v4 CV. Bootstrap compares fixed OOF predictions and does not include refit uncertainty. No competition upload.'),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
