from pathlib import Path
import pandas as pd
import numpy as np
import json, math, itertools, hashlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
RAW=['in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog','out_temp','out_hum','out_rad','out_wspd']
STUDY=ROOT/'집/코덱스/local/statistical_experiments_20261003_v1'

def inputs():
    x=pd.read_csv(ROOT/'공용/대회자료/정형데이터/참가자_배포/train_X.csv',usecols=['row_id']+RAW)
    x=x[x.row_id.str[:3].isin(['F13','F47'])].copy();x['farm']=x.row_id.str[:3];x['day']=x.row_id.str[4:7].astype(int);x['hour']=x.row_id.str[8:10].astype(int)
    grouped=x.groupby(['farm','day'],sort=True)
    f=grouped[RAW].mean().add_suffix('_mean')
    for col in ['in_temp','in_hum','in_co2','out_temp']:f[col+'_std']=grouped[col].std(ddof=0)
    night=x[x.hour<=6].groupby(['farm','day']);day=x[x.hour.between(9,16)].groupby(['farm','day'])
    for col in ['in_temp','in_hum']:f[col+'_day_minus_night']=day[col].mean()-night[col].mean()
    f['vent_zero_fraction']=x.assign(z=x.act_vent.eq(0)).groupby(['farm','day']).z.mean()
    f['night_heating']=night.act_heating.mean()
    h0=x[x.hour==0].set_index(['farm','day'])
    for col in ['act_heating','act_thermal','act_vent','in_temp','in_hum']:f[col+'_h0']=h0[col]
    assert f.shape[1]==27
    return x,f

def outer_members():
    t=pd.read_csv(ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv')
    t=t[(t.validator=='DIAG10')&(t.base_seed==7)&(t.context=='1-8')&t.member.isin(['BASE','CODEX','PFN'])]
    p=t.pivot(index=['farm','day','hour'],columns='member',values='prediction')
    tm=p.groupby(level=[0,1]).mean().std(axis=1,ddof=0);th=p.std(axis=1,ddof=0).groupby(level=[0,1]).mean()
    e=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv');e=e[(e.validator=='DIAG10')&(e.seed==7)]
    m=e.groupby(['farm','day'])[['season_r3','season_pfn']].mean().std(axis=1,ddof=0)
    h=e[['season_r3','season_pfn']].std(axis=1,ddof=0).groupby([e.farm,e.day]).mean()
    folds=e.groupby(['farm','day']).validation_fold.first().to_dict()
    return {'TEMP':(tm,th),'EC':(m,h)},folds

def perm_test(frame,col,labels):
    a=frame[col].to_numpy(float);good=np.asarray(labels,bool)
    if not np.isfinite(a).all() or np.std(a)==0:return dict(p=None,perm_n=0,reason='missing_or_constant')
    observed=abs(float(a[good].mean()-a[~good].mean()))
    strata={}
    for i,r in enumerate(frame.itertuples()):
        direction=int(np.sign(r.gap)) if r.target=='TEMP' else 0
        strata.setdefault((r.farm,int(r.day>=179),direction),[]).append(i)
    blocks=[];total=1
    for ids in strata.values():
        k=int(good[ids].sum());options=list(itertools.combinations(ids,k));blocks.append(options);total*=len(options)
    values=[]
    if total<=20000:
        for choice in itertools.product(*blocks):
            selected=np.array([i for group in choice for i in group],int);mask=np.zeros(len(a),bool);mask[selected]=True
            values.append(abs(float(a[mask].mean()-a[~mask].mean())))
        p=float(np.mean(np.asarray(values)>=observed-1e-12))
    else:
        rng=np.random.default_rng(20261003)
        for _ in range(20000):
            selected=np.array([i for options in blocks for i in options[rng.integers(len(options))]],int);mask=np.zeros(len(a),bool);mask[selected]=True
            values.append(abs(float(a[mask].mean()-a[~mask].mean())))
        p=float((np.sum(np.asarray(values)>=observed-1e-12)+1)/(len(values)+1))
    return dict(p=p,perm_n=len(values),possible_assignments=total,observed_abs_difference=observed)

def main():
    x,f=inputs();members,efolds=outer_members();d=pd.read_csv(ROOT/'집/코덱스/analysis/problem_incidence_20261003_v1/representative_days.csv')
    d=d[((d.target=='EC')&(d.y>=1))|((d.target=='TEMP')&(d.air_n==24)&(d.gap.abs()>=2))].copy()
    all_days=pd.read_csv(ROOT/'집/코덱스/analysis/problem_incidence_20261003_v1/representative_days.csv')
    targets={target:all_days[all_days.target==target].set_index(['farm','day']).y.to_dict() for target in ['TEMP','EC']}
    tfolds={}
    for farm in ['F13','F47']:
        ds=sorted(all_days[(all_days.target=='TEMP')&(all_days.farm==farm)].day.unique())
        tfolds.update({(farm,int(day)):(i//5)%10 for i,day in enumerate(ds)})
    neighbors=[];plans={}
    for target in ['TEMP','EC']:
        for fold in range(10):
            z=np.load(STUDY/f'{"T" if target=="TEMP" else "E"}_DIAG10_{fold}_cpu.npz')
            keys=sorted({(r[:3],int(r[4:7])) for r in z['outer_train_id']})
            tr=f.loc[keys,[col+'_mean' for col in RAW]];med=tr.median();filled=tr.fillna(med);sd=filled.std(ddof=0).replace(0,1)
            plans[(target,fold)]=(keys,tr,med,sd)
    rows=[]
    for row in d.itertuples():
        key=(row.farm,int(row.day));fold=(tfolds if row.target=='TEMP' else efolds)[key];keys,tr,med,sd=plans[(row.target,fold)]
        assert key not in keys
        candidates=[k for k in keys if k[0]==row.farm and (k[1]>=179)==(row.day>=179)]
        v=f.loc[key,[col+'_mean' for col in RAW]].fillna(med)
        dist=np.sqrt((((tr.loc[candidates].fillna(med)-v)/sd)**2).mean(axis=1));order=dist.sort_values().head(3)
        feat=f.loc[key].to_dict();feat['nearest_train_distance']=float(order.iloc[0]);feat['member_day_std']=float(members[row.target][0].loc[key]);feat['member_hour_std']=float(members[row.target][1].loc[key])
        good=row.rmse<=(.1 if row.target=='EC' else .5);relaxed=row.rmse<=(.15 if row.target=='EC' else .7)
        rows.append(dict(target=row.target,farm=row.farm,day=int(row.day),fold=int(fold),good=good,relaxed_good=relaxed,y=row.y,pred=row.pred,gap=row.gap,bias=row.bias,rmse=row.rmse,**feat))
        for rank,(nk,value) in enumerate(order.items(),1):neighbors.append(dict(target=row.target,farm=row.farm,day=int(row.day),good=good,rank=rank,neighbor_farm=nk[0],neighbor_day=int(nk[1]),distance=float(value),neighbor_truth=targets[row.target].get(nk),truth_difference=targets[row.target].get(nk,float('nan'))-row.y))
    days=pd.DataFrame(rows);features=list(f.columns)+['nearest_train_distance','member_day_std','member_hour_std'];assert len(features)==30
    comparisons=[];matches=[];severity=[]
    for target,q in days.groupby('target',sort=True):
        q=q.reset_index(drop=True);positive=q[q.good];negative=q[~q.good]
        severity.append(dict(target=target,n=len(q),good_n=len(positive),good_truth=positive.y.tolist(),good_gap=positive.gap.tolist(),negative_truth_mean=float(negative.y.mean()),negative_abs_gap_mean=float(negative.gap.abs().mean()) if target=='TEMP' else None))
        for col in features:
            good=positive[col].to_numpy();bad=negative[col].to_numpy();mean=float(np.nanmean(good)-np.nanmean(bad));percentiles=[float(np.mean(bad<=v)) if np.isfinite(v) else None for v in good];relaxed=q[q.relaxed_good];rbad=q[~q.relaxed_good]
            statistic=perm_test(q,col,q.good)
            comparisons.append(dict(target=target,feature=col,good_n=len(good),bad_n=len(bad),good_values=good.tolist(),good_mean=float(np.nanmean(good)),bad_mean=float(np.nanmean(bad)),difference=mean,standardized_difference=mean/float(np.nanstd(q[col])) if np.nanstd(q[col])>0 else None,good_bad_percentiles=percentiles,all_good_high_quartile=all(v is not None and v>=.75 for v in percentiles),all_good_low_quartile=all(v is not None and v<=.25 for v in percentiles),relaxed_good_n=len(relaxed),relaxed_difference=float(relaxed[col].mean()-rbad[col].mean()),p_bonferroni=min(1,statistic['p']*60) if statistic['p'] is not None else None,**statistic))
        for good in positive.itertuples():
            available=negative[(negative.farm==good.farm)&((negative.day>=179)==(good.day>=179))].copy()
            if target=='TEMP':available=available[np.sign(available.gap)==np.sign(good.gap)];available['severity_delta']=(available.gap-good.gap).abs()
            else:available['severity_delta']=(available.y-good.y).abs()
            chosen=available.sort_values(['severity_delta','day']).head(3)
            for col in features:
                matches.append(dict(target=target,farm=good.farm,day=int(good.day),feature=col,match_n=len(chosen),match_days=chosen.day.astype(int).tolist(),severity_deltas=chosen.severity_delta.tolist(),good_value=float(getattr(good,col)),matched_bad_mean=float(chosen[col].mean()) if len(chosen) else None,delta=float(getattr(good,col)-chosen[col].mean()) if len(chosen) else None))
    days.to_csv(HERE/'event_days.csv',index=False);pd.DataFrame(neighbors).to_csv(HERE/'neighbors.csv',index=False);pd.DataFrame(comparisons).to_csv(HERE/'comparisons.csv',index=False);pd.DataFrame(matches).to_csv(HERE/'matches.csv',index=False)
    result=dict(status='EXPLORATORY',features=features,family=60,severity=severity,comparisons=comparisons,matches=matches,neighbors=neighbors,code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    out=pd.DataFrame(comparisons)
    print(out.sort_values('p').head(16)[['target','feature','good_mean','bad_mean','difference','p','p_bonferroni','all_good_high_quartile','all_good_low_quartile','relaxed_difference']].to_string(index=False))
    print('OVERLAPPING QUARTILES');print(out[out.all_good_high_quartile|out.all_good_low_quartile][['target','feature','good_values','bad_mean','p','p_bonferroni','relaxed_difference']].to_string(index=False))
    print('SEVERITY',json.dumps(severity));print('GOOD DAYS');print(days[days.good][['target','farm','day','y','gap','rmse','act_vent_mean','in_hum_mean','nearest_train_distance','member_day_std']].to_string(index=False))

if __name__=='__main__':main()
