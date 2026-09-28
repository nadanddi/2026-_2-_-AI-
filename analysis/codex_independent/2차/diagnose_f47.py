# -*- coding: utf-8 -*-
import sys,json
sys.dont_write_bytecode=True
from evaluate_task1 import *
d=pd.read_json(H/'validation_rows.json'); pack=np.load(H/'validation_predictions.npz',allow_pickle=True)
assert np.array_equal(pack['row_id'],d.row_id)
saved=np.load(ROOT/'research/local/eval_v6_oof.npz',allow_pickle=True)
b=pd.Series(saved['F60ND__DIAG10'],index=saved['row_id']).loc[d.row_id].to_numpy()
tx,ty,sx=common.load_raw(); aa=pd.concat([tx,sx],ignore_index=True).sort_values(['farm','t']).reset_index(drop=True)
prev=aa.groupby('farm').in_temp.shift().where(aa.groupby('farm').t.diff()==1)
aa['jump']=(aa.in_temp-prev).where(aa.hour==0)
jump=aa[aa.hour==0].set_index(['farm','day']).jump
result={'variants':{}}
for v in ['unweighted','weighted','weighted_foldlocal']:
    p=.8*b+.2*pack['DIAG10__'+v]
    q=d.copy(); q['eb']=b-q.sub_temp; q['ep']=p-q.sub_temp; q['dsse']=q.ep**2-q.eb**2
    focus=(q.farm=='F47')&(q.day>=179)
    f=q[focus].copy()
    rows=[]
    for (farm,day),g in f.groupby(['farm','day']):
        rows.append({'farm':farm,'day':int(day),'n':len(g),'rmse_base':float(np.sqrt(np.mean(g.eb**2))),'rmse_mix':float(np.sqrt(np.mean(g.ep**2))),'delta_sse':float(g.dsse.sum()),'bias_base':float(g.eb.mean()),'bias_mix':float(g.ep.mean()),'early_delta_sse':float(g.loc[g.hour<4,'dsse'].sum()),'late_delta_sse':float(g.loc[g.hour>=4,'dsse'].sum()),'in_temp_mean':float(g.in_temp.mean()),'in_temp_min':float(g.in_temp.min()),'ph3_min':float(g.day_ph3_min.iloc[0]),'jump':float(jump.get((farm,day),np.nan)),'heat_mean':float(g.act_heating.mean()),'fan_mean':float(g.act_circfan.mean()),'vent_zero_share':float((g.act_vent==0).mean()),'noise_score':float(g.noise_score.iloc[0]),'noisy':bool(g.noisy.iloc[0]),'low_noise':bool(g.low_noise.iloc[0]),'clean_rows':int(g.clean.sum())})
    daytab=pd.DataFrame(rows).sort_values('delta_sse',ascending=False)
    hr={str(h):comparison(f.loc[f.hour==h].reset_index(drop=True),b[focus][f.hour==h],p[focus][f.hour==h]) for h in range(24)}
    subs={}
    for nm,m in [('all',focus),('cold10',focus&(q.day_ph3_min<10)),('not_cold10',focus&(q.day_ph3_min>=10)),('clean_low_noise',focus&q.clean&q.low_noise),('jump3',focus&q.set_index(['farm','day']).index.map(jump).to_series(index=q.index).abs().ge(3))]:
        subs[nm]=comparison(q.loc[m].reset_index(drop=True),b[m],p[m])
    for k in [1,3]:
        worst=daytab.head(k).day.to_list();m=focus&~q.day.isin(worst)
        subs['exclude_worst'+str(k)]=comparison(q.loc[m].reset_index(drop=True),b[m],p[m])
    result['variants'][v]={'days':daytab.to_dict('records'),'hours':hr,'subsets':subs,'worse_days':int((daytab.delta_sse>0).sum()),'all_delta_sse':float(daytab.delta_sse.sum()),'positive_delta_sse':float(daytab.loc[daytab.delta_sse>0,'delta_sse'].sum()),'top3_positive_share':float(daytab.head(3).delta_sse.sum()/daytab.loc[daytab.delta_sse>0,'delta_sse'].sum()),'correlations':{c:float(daytab[c].corr(daytab.delta_sse,method='spearman')) for c in ['in_temp_mean','ph3_min','heat_mean','jump','noise_score']}}
    print(v,'summary',{k:val for k,val in result['variants'][v].items() if k not in ['days','hours']},flush=True)
    print('topdays',daytab.head(5).to_json(orient='records'),flush=True)
(H/'f47_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
