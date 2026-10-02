from support import *
def main():
    groups=pd.read_csv(ROOT/'집/코덱스/analysis/temp_validation_structure_20261003_v1/weather_groups.csv');lookup={(f,int(d)):int(g) for f,d,g in groups.itertuples(index=False,name=None)}
    tp=pd.read_csv(ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv',usecols=['row_id','farm','day','validator','base_seed','context','member']);td=tp[(tp.validator=='DIAG10')&(tp.base_seed==7)&(tp.context=='1-8')&(tp.member=='W30G')];tfs=diag_folds(td)
    ep=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv',usecols=['row_id','validator','validation_fold','seed']);records=[]
    def summary(train,query):
        tk={(r[:3],int(r[4:7])) for r in train};qk={(r[:3],int(r[4:7])) for r in query};tg={lookup[k] for k in tk};shared=sum(lookup[k] in tg for k in qk);return dict(train_days=len(tk),query_days=len(qk),train_weather_groups=len(tg),shared_query_days=shared,shared_query_pct=100*shared/len(qk))
    for cp in OUT.glob('*_cpu.npz'):
        prefix=cp.name.split('_cpu')[0];target,name,k=prefix.split('_');k=int(k);z=dict(np.load(cp))
        if target=='T':
            q=td[[int(d) in tfs[k][f] for f,d in zip(td.farm,td.day)]] if name=='DIAG10' else tp[(tp.validator==name)&(tp.base_seed==7)&(tp.context=='1-8')&(tp.member=='W30G')]
        else:q=ep[(ep.validator==name)&(ep.validation_fold==k)&(ep.seed==7)]
        records.append(dict(prefix=prefix,scope='outer',**summary(z['outer_train_id'],q.row_id)));records.append(dict(prefix=prefix,scope='inner_actual',**summary(z['inner_train_id'],z['row_id'])))
    assert len(records)==68;pd.DataFrame(records).to_csv(HERE/'weather_audit.csv',index=False);savej(HERE/'weather_audit.json',dict(status='AUDIT_ONLY',cells=len(records),global_train_weather_group_definition='preexisting audit only, never model features',records=records,source_hash=sha(__file__)));print('WEATHER_AUDIT_DONE',flush=True)
if __name__=='__main__':main()
