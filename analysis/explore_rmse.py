"""Bounded input-only ablation search; confirmation never selects candidates."""
from train_initial_model import *
from hypothesis_features import build as hypothesis_build

DEST=ROOT/'analysis/local/rmse_exploration'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    x=load('train_X.csv');x=x[x.farm.isin(FARMS)]
    f,g=build(x);hf,_=hypothesis_build(x)
    meta=x.set_index('row_id').loc[f.index];y=load('train_y.csv').set_index('row_id').reindex(f.index)
    extra=[c for c in hf if c not in f]
    f=hf
    configs={}
    for target in TARGETS:
        base=g[target] if target=='sub_temp' else g['raw']
        configs[target]={
            'baseline':(base,{},False),
            'regularized':(base,dict(max_leaf_nodes=7,min_samples_leaf=80,max_iter=240),False),
            'flexible':(base,dict(max_leaf_nodes=23,min_samples_leaf=20,max_iter=300),False),
            'no_day':([c for c in base if c!='day'],{},False),
            'day_context':(base+[c for c in extra if c.endswith(('_hyp_start','_hyp_today_mean'))],{},False),
            'lag48':(base+[c for c in extra if c.endswith(('_hyp_lag48','_hyp_delta48'))],{},False),
            'bridge':(base+[c for c in extra if '_hyp_bridge' in c],{},False),
        }
        if target=='sub_temp':configs[target]['air_residual']=(base,{},True)
        else:configs[target]['compact_regularized']=(g[target],dict(max_leaf_nodes=7,min_samples_leaf=80,max_iter=240),False)
    records=[];preds=[];chosen={}
    def evaluate(a,b,phase,mode):
        for farm in FARMS:
            local=meta.farm.eq(farm);sig=signatures(meta[local].reset_index())
            valid=local&meta.day.ge(a)&meta.day.lt(b)
            reserved=local&meta.day.ge(CHECK[0])&meta.day.lt(CHECK[1])
            forbidden=sig.reindex(meta.loc[valid|reserved,'day'].unique()).dropna()
            tr=local&~(valid|reserved)&~meta.day.isin(sig.index[sig.isin(forbidden)])
            if mode=='past_only':tr &= meta.day.lt(a)
            assert not (tr&valid).any()
            for target in TARGETS:
                names=list(configs[target]) if phase=='selection' else list(dict.fromkeys(['baseline',chosen[target]]))
                for name in names:
                    cols,override,residual=configs[target][name]
                    offset=f.in_temp_mean3.fillna(f.out_temp).fillna(0) if residual else pd.Series(0.,index=f.index)
                    model=HistGradientBoostingRegressor(**(PARAMS|override))
                    model.fit(f.loc[tr,cols],y.loc[tr,target]-offset[tr])
                    p=model.predict(f.loc[valid,cols])+offset[valid].to_numpy()
                    records.append(dict(phase=phase,mode=mode,farm=farm,start=a,target=target,variant=name,**score(y.loc[valid,target],p)))
                    preds.extend(dict(phase=phase,mode=mode,row_id=rid,target=target,variant=name,actual=float(v),prediction=float(pv)) for rid,v,pv in zip(f.index[valid],y.loc[valid,target],p))
            print(f'{phase} {mode} {farm} {a}:{b} finished',flush=True)
        pd.DataFrame(records).to_csv(DEST/'scores.csv',index=False)
    for a,b in FOLDS:evaluate(a,b,'selection','blocked')
    def aggregate():
        d=pd.DataFrame(records);d['sse']=d.rmse**2*d.n
        s=d.groupby(['phase','mode','target','variant']).agg(sse=('sse','sum'),n=('n','sum'))
        s['rmse']=np.sqrt(s.sse/s.n)
        return s.drop(columns='sse').reset_index()
    s=aggregate()
    chosen={t:s[s.target.eq(t)].sort_values('rmse').iloc[0]['variant'] for t in TARGETS}
    print('Selected only on development:',chosen,flush=True)
    evaluate(*CHECK,'confirmation','blocked');evaluate(*CHECK,'confirmation','past_only')
    s=aggregate();s.to_csv(DEST/'summary.csv',index=False)
    pd.DataFrame(preds).to_csv(DEST/'predictions.csv',index=False)
    # Verify unchanged baseline matches the earlier experiment.
    old=pd.read_csv(ROOT/'analysis/local/initial_model_v1/validation_summary.csv')
    for r in s[s.variant.eq('baseline')].itertuples():
        reference=old[(old.phase==r.phase)&(old['mode']==r.mode)&(old.target==r.target)&(old.variant==('compact' if r.target=='sub_temp' else 'raw'))]
        assert len(reference)==1 and abs(reference.iloc[0].rmse-r.rmse)<1e-10
    (DEST/'report.json').write_text(json.dumps(dict(chosen=chosen,configs=configs,params=PARAMS,folds=FOLDS,confirmation=CHECK,limitations='Previously examined periods; exploratory validation, not an unseen benchmark. No test labels or future features.',summary=s.to_dict('records')),ensure_ascii=False,indent=2),encoding='utf-8')
    print(s.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
