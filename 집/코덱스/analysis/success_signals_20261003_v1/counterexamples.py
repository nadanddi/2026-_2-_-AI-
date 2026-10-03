from diagnose import *

def main():
    d=pd.read_csv(HERE/'event_days.csv');records=[]
    for target,col in [('TEMP','out_temp_std'),('EC','in_temp_day_minus_night')]:
        q=d[d.target==target];threshold=float(q[q.good][col].min())
        counter=q[(~q.good)&(q[col]>=threshold-1e-12)]
        records.append(dict(target=target,feature=col,criterion='at least the minimum observed successful value; posthoc descriptive counterexample',threshold=threshold,bad_days=counter[['farm','day','y','gap','rmse',col]].to_dict(orient='records')))
    x=pd.read_csv(ROOT/'공용/대회자료/정형데이터/참가자_배포/train_X.csv',usecols=['row_id','out_temp','out_hum','out_rad','out_wspd'])
    a=x[x.row_id.str.startswith('F13_194_')].sort_values('row_id');b=x[x.row_id.str.startswith('F47_191_')].sort_values('row_id')
    cols=['out_temp','out_hum','out_rad','out_wspd'];matrix=a[cols].to_numpy()-b[cols].to_numpy()
    assert len(a)==len(b)==24 and np.isfinite(matrix).all()
    maxdiff=float(np.max(np.abs(matrix)));scalar=max(abs(float(v)-float(w)) for col in cols for v,w in zip(a[col],b[col]));assert maxdiff==scalar==0
    cases=d[(d.target=='TEMP')&(((d.farm=='F13')&(d.day==194))|((d.farm=='F47')&(d.day==191)))][['farm','day','good','y','pred','gap','rmse']].to_dict(orient='records')
    result=dict(status='PASS',counterexamples=records,weather_twin=dict(columns=cols,hours=24,max_abs_difference=maxdiff,cases=cases),method='numpy 24x4 matrix versus independent scalar max',scope='posthoc descriptive counterexamples, no new selection or significance test')
    (HERE/'counterexamples.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
