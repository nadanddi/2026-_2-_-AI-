from pathlib import Path
import json, math, hashlib
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def load():
    t=pd.read_csv(ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv')
    t=t[t.member=='W30G'].copy()
    x=pd.read_csv(ROOT/'공용/대회자료/정형데이터/참가자_배포/train_X.csv',usecols=['row_id','in_temp'])
    t=t.merge(x,on='row_id',validate='many_to_one')
    t=t.rename(columns={'prediction':'pred','sub_temp':'y','base_seed':'seed'})
    e=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv')
    e=e.rename(columns={'season_v2':'pred','sub_ec':'y'})
    e['context']='1-4'
    return t,e

def daily(q,target):
    q=q.copy();q['se']=(q.pred-q.y)**2
    g=q.groupby(['farm','day'],sort=True).agg(y=('y','mean'),pred=('pred','mean'),n=('y','size'),sse=('se','sum'))
    if target=='TEMP':
        air=q.groupby(['farm','day']).in_temp.agg(['mean','count'])
        g['air']=air['mean'];g['air_n']=air['count'];g['gap']=g.y-g.air
    g['bias']=g.pred-g.y;g['rmse']=np.sqrt(g.sse/g.n)
    return g

def summarize(g,target,threshold):
    if target=='EC':
        event=g.y>=threshold
        failure=event&(g.bias<=-.2)
        good=event&(g.rmse<=.1)
    else:
        event=(g.gap.abs()>=threshold)&(g.air_n==24)
        failure=event&(g.bias*g.gap<0)&(g.bias.abs()>=.5)
        good=event&(g.rmse<=.5)
    d=g[event]; f=g[failure]; ok=g[good]
    return dict(days=len(g),rows=int(g.n.sum()),event_days=int(event.sum()),event_pct=100*event.mean(),
                event_rows=int(d.n.sum()),event_sse_pct=100*d.sse.sum()/g.sse.sum(),
                failure_days=int(failure.sum()),failure_pct_all=100*failure.mean(),failure_sse_pct=100*f.sse.sum()/g.sse.sum(),
                good_days=int(good.sum()),good_pct_event=100*good.sum()/event.sum() if event.any() else None,
                event_rmse=float(np.sqrt(d.sse.sum()/d.n.sum())) if len(d) else None,
                mean_truth=float(d.y.mean()) if len(d) else None,mean_prediction=float(d.pred.mean()) if len(d) else None,
                incomplete_air_days=int((g.air_n!=24).sum()) if target=='TEMP' else None)

def main():
    t,e=load();records=[];examples=[];daily_records=[]
    for target,frame in [('TEMP',t),('EC',e)]:
        for (val,seed,context),q in frame.groupby(['validator','seed','context'],sort=True):
            g=daily(q,target)
            for threshold in ([1.,2.,3.] if target=='TEMP' else [1.]):
                records.append(dict(target=target,validator=val,seed=int(seed),context=context,threshold=threshold,segment='all',**summarize(g,target,threshold)))
            threshold=2. if target=='TEMP' else 1.
            for segment,mask in [('F13',g.index.get_level_values(0)=='F13'),('F47',g.index.get_level_values(0)=='F47'),('early',g.index.get_level_values(1)<179),('late',g.index.get_level_values(1)>=179)]:
                records.append(dict(target=target,validator=val,seed=int(seed),context=context,threshold=threshold,segment=segment,**summarize(g[mask],target,threshold)))
            if val=='DIAG10' and int(seed)==7 and context==('1-8' if target=='TEMP' else '1-4'):
                daily_records.append(g.reset_index().assign(target=target))
                categories=[('high',g.y>=1)] if target=='EC' else [('warmer',(g.gap>=2)&(g.air_n==24)),('cooler',(g.gap<=-2)&(g.air_n==24))]
                for category,mask in categories:
                    for quality,rows in [('best',g[mask].sort_values('rmse').head(3)),('worst',g[mask].sort_values('rmse').tail(3))]:
                        for (farm,day),row in rows.iterrows():
                            examples.append(dict(target=target,category=category,quality=quality,farm=farm,day=int(day),**{k:float(v) for k,v in row.items()}))
                if target=='TEMP':
                    for category,mask in categories:
                        h=g.copy();h.loc[~mask,'gap']=0.
                        records.append(dict(target=target,validator=val,seed=int(seed),context=context,threshold=2.,segment=category,**summarize(h,target,2.)))
    pd.DataFrame(records).to_csv(HERE/'incidence.csv',index=False)
    pd.DataFrame(examples).to_csv(HERE/'examples.csv',index=False)
    pd.concat(daily_records,ignore_index=True).to_csv(HERE/'representative_days.csv',index=False)
    result=dict(status='EXPLORATORY',records=records,examples=examples,code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(records).query("validator=='DIAG10' and seed==7 and segment in ['all','warmer','cooler'] and ((target=='TEMP' and context=='1-8') or target=='EC')").to_string(index=False))

if __name__=='__main__':main()
