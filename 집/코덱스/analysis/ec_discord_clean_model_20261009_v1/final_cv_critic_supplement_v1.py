from pathlib import Path
import csv,json,math,statistics
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'집/코덱스/local'/H.name
D=json.loads((H/'final_independent_v1.json').read_text(encoding='utf8'));G={(r['scope'],r['arm'],r['seed']):r for r in D['groups']}
rows=[];train=[]
for k in range(10):
    with (L/'cv_predictions'/f'fold{k}.csv').open(encoding='utf8',newline='') as f:rows.extend(csv.DictReader(f))
    m=json.loads((L/'cv_predictions'/f'fold{k}.json').read_text(encoding='utf8'));assert m['candidate_models']==9;train.append(dict(k=k,rows=len(m['train_ids']),rmse=m['clean_train_rmse']['ensemble']))
def rmse(rr):return math.sqrt(math.fsum((float(r['prediction'])-float(r['sub_ec']))**2 for r in rr)/len(rr))
ens=[r for r in rows if r['seed']=='ensemble'];days=defaultdict(list)
for r in ens:
    if r['arm']=='BASE':days[r['farm'],int(r['day'])].append(float(r['sub_ec']))
high={k:math.fsum(v)/24>=1 for k,v in days.items()}
strata=[]
for farm in ('F13','F47'):
    for p2 in (False,True):
        for kind in ('all','filtered','removed','normal','high','retained_high'):
            d={}
            for a in ('BASE','CLEAN'):
                rr=[r for r in ens if r['arm']==a and r['farm']==farm and (int(r['day'])>=179)==p2]
                if kind=='filtered':rr=[r for r in rr if r['query_removed']=='False']
                if kind=='removed':rr=[r for r in rr if r['query_removed']=='True']
                if kind=='normal':rr=[r for r in rr if not high[r['farm'],int(r['day'])]]
                if kind=='high':rr=[r for r in rr if high[r['farm'],int(r['day'])]]
                if kind=='retained_high':rr=[r for r in rr if high[r['farm'],int(r['day'])] and r['query_removed']=='False']
                if rr:d[a]=dict(rmse=rmse(rr),rows=len(rr),days=len({r['day'] for r in rr}))
            if d:strata.append(dict(farm=farm,pass_number=2 if p2 else 1,scope=kind,**d,pct=100*(d['CLEAN']['rmse']/d['BASE']['rmse']-1)))
folds=[]
for a in ('BASE','CLEAN'):
    for sc in ('all','filtered'):
        v=[]
        for k in range(10):
            rr=[r for r in ens if int(r['k'])==k and r['arm']==a and (sc=='all' or r['query_removed']=='False')];v.append(rmse(rr))
        folds.append(dict(arm=a,scope=sc,fold_mean=statistics.mean(v),fold_sd=statistics.stdev(v),min=min(v),max=max(v),fold_rmse=v))
concentration=[]
for sc in ('all','filtered','removed','normal','high','retained_high'):
    by=defaultdict(dict)
    for r in ens:
        ishigh=high[r['farm'],int(r['day'])];removed=r['query_removed']=='True'
        if not(sc=='all' or sc=='filtered' and not removed or sc=='removed' and removed or sc=='normal' and not ishigh or sc=='high' and ishigh or sc=='retained_high' and ishigh and not removed):continue
        by[r['farm'],int(r['day'])].setdefault(r['arm'],[]).append((float(r['prediction'])-float(r['sub_ec']))**2)
    v=[dict(farm=f,day=d,benefit_sse=math.fsum(q['BASE'])-math.fsum(q['CLEAN'])) for (f,d),q in by.items()]
    pos=sorted((r for r in v if r['benefit_sse']>0),key=lambda r:r['benefit_sse'],reverse=True);neg=sorted((r for r in v if r['benefit_sse']<0),key=lambda r:r['benefit_sse'])
    gain=math.fsum(r['benefit_sse'] for r in pos);loss=-math.fsum(r['benefit_sse'] for r in neg)
    concentration.append(dict(scope=sc,days=len(v),better_days=len(pos),worse_days=len(neg),net_benefit_sse=gain-loss,gross_gain_sse=gain,gross_loss_sse=loss,top5_gain_share=math.fsum(r['benefit_sse'] for r in pos[:5])/gain if gain else None,top5_loss_share=-math.fsum(r['benefit_sse'] for r in neg[:5])/loss if loss else None,top5_gains=pos[:5],top5_losses=neg[:5]))
changes=[]
for sc in {k[0] for k in G}:
    for s in ('7','101','2024','ensemble'):
        if (sc,'BASE',s) in G:changes.append(dict(scope=sc,seed=s,baseline=G[sc,'BASE',s]['rmse'],clean=G[sc,'CLEAN',s]['rmse'],pct=100*(G[sc,'CLEAN',s]['rmse']/G[sc,'BASE',s]['rmse']-1)))
out=dict(status='FINAL_CV_SUPPLEMENT_PASS',strata=strata,fold_variation=folds,train_rmse_receipts=train,train_rmse_mean=statistics.mean(r['rmse'] for r in train),train_rmse_min=min(r['rmse'] for r in train),train_rmse_max=max(r['rmse'] for r in train),changes=changes,concentration=concentration,cv_models=90,baseline_replay_models=3,final_models_pending=True,adoption=False)
with (H/'final_cv_critic_supplement_v1.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'fold_variation':folds,'train_mean_min_max':[out['train_rmse_mean'],out['train_rmse_min'],out['train_rmse_max']],'changes':[r for r in changes if r['seed']=='ensemble'],'strata':strata,'concentration':concentration},ensure_ascii=False,indent=2))
