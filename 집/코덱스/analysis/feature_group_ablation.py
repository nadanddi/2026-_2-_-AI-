"""Retrained feature-group ablation for v4 with farm-day cluster bootstrap CIs."""
from compare_lgbm_extratrees import *

OUTA=ROOT/'analysis/local/feature_group_ablation_v4'
BOOTSTRAPS=5000

def group_for(c):
    # Mutually exclusive groups; missing/coverage indicators are evaluated together.
    if c.endswith(('_absent','_missing','_age','_coverage6')):return 'data_quality'
    if c=='farm_code':return 'farm_identity'
    if c in ['day','hour_sin','hour_cos','midnight']:return 'time'
    if c.startswith('in_temp') or c in ['temp_difference','vent_temp']:return 'internal_temperature'
    if c.startswith('in_hum'):return 'internal_humidity'
    if c.startswith('in_co2'):return 'internal_co2'
    if c.startswith('out_') or c.startswith('rad_') or c in ['in_rad','vent_wind']:return 'external_radiation'
    if c.startswith('act_') or c=='heating_closed':return 'actuators'
    raise AssertionError(f'Unassigned feature: {c}')

def cluster_bootstrap(actual,baseline,candidate,clusters,seed):
    d=pd.DataFrame(dict(y=actual,b=baseline,c=candidate,cluster=clusters))
    a=d.groupby('cluster').agg(n=('y','size'),base=('b',lambda z:0.),cand=('c',lambda z:0.))
    # Aggregate squared errors explicitly to retain RMSE scale under resampling.
    d['base_se']=(d.y-d.b)**2;d['cand_se']=(d.y-d.c)**2
    a=d.groupby('cluster').agg(n=('y','size'),base=('base_se','sum'),cand=('cand_se','sum'))
    arr=a[['n','base','cand']].to_numpy();rng=np.random.default_rng(seed)
    values=np.empty(BOOTSTRAPS)
    for start in range(0,BOOTSTRAPS,250):
        size=min(250,BOOTSTRAPS-start);idx=rng.integers(0,len(arr),size=(size,len(arr)))
        sampled=arr[idx].sum(axis=1)
        values[start:start+size]=np.sqrt(sampled[:,2]/sampled[:,0])-np.sqrt(sampled[:,1]/sampled[:,0])
    return np.quantile(values,[.025,.5,.975])

def main():
    OUTA.mkdir(parents=True,exist_ok=True)
    x=load('train_X.csv');y=load('train_y.csv').set_index('row_id')
    f,g=features(x);m=x.set_index('row_id').loc[f.index];labels=y.reindex(f.index)
    assignment,splits=folds(x);assignment=assignment.reindex(f.index)
    groups=sorted({group_for(c) for c in set(g['full'])|set(g['separate_temp'])|set(g['ec'])})
    mapping=pd.DataFrame([(scope,c,group_for(c)) for scope in ['full','separate_temp','ec'] for c in g[scope]],columns=['scope','feature','group'])
    mapping.to_csv(OUTA/'feature_groups.csv',index=False)
    base=pd.read_csv(ROOT/'analysis/local/lgbm_extratrees_v4/oof_predictions.csv')
    base=base[((base.target=='sub_temp')&base.variant.eq('blend_50'))|((base.target=='sub_ec')&base.variant.eq('leaf_2'))]
    base=base.set_index(['target','row_id'])
    predictions=[]
    checkpoint=OUTA/'ablation_predictions.csv'
    done=set()
    if checkpoint.exists():
        previous=pd.read_csv(checkpoint);predictions=previous.to_dict('records')
        done=set(zip(previous.target,previous.group,previous.fold))
    for k,(tr0,va0) in enumerate(splits):
        for target in TARGETS:
            mask=m.farm.isin(FARMS)&labels[target].notna();valid=va0.reindex(f.index)&mask
            for removed in groups:
                if (target,removed,k) in done:continue
                total=pd.Series(0.,index=f.index[valid])
                parts=[('separate_temp',False,1,.5),('full',True,20,.5)] if target=='sub_temp' else [('ec',True,1,1.)]
                for scope,pooled,weight,coef in parts:
                    for farm in ([None] if pooled else FARMS):
                        tr=tr0.reindex(f.index)&labels[target].notna();va=valid.copy()
                        if farm is not None:tr &= m.farm.eq(farm);va &= m.farm.eq(farm)
                        cols=[c for c in g[scope] if group_for(c)!=removed and f.loc[tr,c].notna().any()]
                        model=fit(target,'leaf_2',f.loc[tr,cols],labels.loc[tr,target],np.where(m.loc[tr,'farm'].isin(FARMS),weight,1.))
                        total.loc[f.index[va]]+=coef*model.predict(f.loc[va,cols])
                predictions.extend(dict(row_id=rid,target=target,group=removed,fold=k,actual=float(labels.at[rid,target]),prediction=float(total[rid])) for rid in total.index)
                pd.DataFrame(predictions).to_csv(checkpoint,index=False)
                print(f'fold {k+1}/5 {target} remove {removed}',flush=True)
    p=pd.DataFrame(predictions);rows=[]
    for (target,removed),q in p.groupby(['target','group']):
        ref=base.loc[target].loc[q.row_id]
        np.testing.assert_allclose(ref.actual,q.actual,rtol=0,atol=1e-12)
        baseline=score(q.actual,ref.prediction.to_numpy())['rmse'];candidate=score(q.actual,q.prediction)['rmse']
        mm=m.loc[q.row_id];clusters=mm.farm+'_'+mm.day.astype(str)
        ci=cluster_bootstrap(q.actual.to_numpy(),ref.prediction.to_numpy(),q.prediction.to_numpy(),clusters.to_numpy(),20260919+len(rows))
        fold_deltas=[]
        for k in range(5):
            z=q.fold.eq(k);fold_deltas.append(score(q.loc[z,'actual'],q.loc[z,'prediction'].to_numpy())['rmse']-score(q.loc[z,'actual'],ref.loc[q.loc[z,'row_id'],'prediction'].to_numpy())['rmse'])
        rows.append(dict(target=target,removed_group=removed,n=len(q),baseline_rmse=baseline,ablated_rmse=candidate,
                         delta_rmse=candidate-baseline,ci_low=ci[0],bootstrap_median=ci[1],ci_high=ci[2],
                         folds_worse=sum(v>0 for v in fold_deltas),fold_deltas=';'.join(f'{v:.8f}' for v in fold_deltas)))
    result=pd.DataFrame(rows).sort_values(['target','delta_rmse'],ascending=[True,False]);result.to_csv(OUTA/'summary.csv',index=False)
    (OUTA/'report.json').write_text(json.dumps(dict(bootstrap_repetitions=BOOTSTRAPS,cluster='farm-day',groups=groups,results=result.to_dict('records'),
        interpretation='Positive delta means removing the group worsened RMSE. CI excludes zero only as stability evidence under farm-day resampling of fixed OOF predictions; it is not a classical feature p-value and does not include model-refit uncertainty.',
        limitation='Same v4 CV reused after prior model selection. Groups remove correlated features together but still overlap information conceptually. Multiple comparisons are descriptive; no causal claim.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
