from pathlib import Path
import csv,json,math,statistics,hashlib
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'집/코덱스/local'/H.name
D=json.loads((H/'final_independent_v5.json').read_text(encoding='utf8'));G={(r['scope'],r['arm'],r['seed']):r for r in D['groups']}
A=('ALL_HIGH','DISCORD_HIGH','CONTROL');S=('7','101','2024');alpha=.025/3
changes=[]
for scope in sorted({k[0] for k in G}):
    for a in A:
        for s in S+('ensemble',):
            k=(scope,a,s);b=(scope,'BASE',s)
            if k in G and b in G:
                changes.append(dict(scope=scope,arm=a,seed=s,rows=G[k]['rows'],days=G[k]['days'],rmse=G[k]['rmse'],baseline=G[b]['rmse'],pct=100*(G[k]['rmse']/G[b]['rmse']-1),raw_et_pct=100*(G[k]['raw_et_rmse']/G[b]['raw_et_rmse']-1)))
seedbetter={a:all(G['normal',a,s]['rmse']<G['normal','BASE',s]['rmse'] for s in S) for a in A}
better_control=all(G['normal','DISCORD_HIGH',s]['rmse']<G['normal','CONTROL',s]['rmse'] for s in S)
p={(b['arm'],b['reference']):b['p_worse'] for b in D['bootstrap']}
screen={'ALL_HIGH':seedbetter['ALL_HIGH'] and p['ALL_HIGH','BASE']<alpha,'DISCORD_HIGH':seedbetter['DISCORD_HIGH'] and better_control and p['DISCORD_HIGH','BASE']<alpha and p['DISCORD_HIGH','CONTROL']<alpha}
guard={a:any(G[sc,a,s]['rmse']/G[sc,'BASE',s]['rmse']>=1.02 for sc in ('all','pass2_normal') for s in S+('ensemble',)) for a in A}
foldvar=[]
for a in ('BASE',)+A:
    for s in S+('ensemble',):
        v=[r['rmse'] for r in D['fold_normal'] if r['arm']==a and r['seed']==s]
        foldvar.append(dict(arm=a,seed=s,folds=len(v),mean=statistics.mean(v),sample_sd=statistics.stdev(v),min=min(v),max=max(v)))
ens=defaultdict(dict);fit=0;maxgap=0;receipts=[]
for k in range(10):
    m=json.loads((L/'folds'/f'fold{k}.json').read_text(encoding='utf8'));fit+=sum(x['new_fit'] for x in m['fitinfo']);maxgap=max(maxgap,m['baseline_final_maxdiff'])
    assert len(m['fitinfo'])==12 and sum(x['new_fit'] for x in m['fitinfo'])==9
    with (L/'folds'/f'fold{k}.csv').open(encoding='utf8',newline='') as f:
        for r in csv.DictReader(f):
            if r['seed']=='ensemble':ens[r['arm']][r['row_id']]=r
    receipts.append(dict(k=k,new_candidate_fits=9,baseline_maxdiff=m['baseline_final_maxdiff']))
assert fit==90 and all(len(ens[a])==8640 for a in ('BASE',)+A)
for part in ('early','late'):
    assert not (H/f'worker_{part}_v1.lock').exists()
    assert f'PART_COMPLETE {part}' in (H/f'{part}_v1.log').read_text(encoding='utf8')
byday=defaultdict(list)
for rid,r in ens['BASE'].items():byday[(r['farm'],int(r['day']))].append(rid)
hi={d:math.fsum(float(ens['BASE'][rid]['sub_ec']) for rid in ids)/24>=1 for d,ids in byday.items()}
conc=[]
for a in A:
    vals=[]
    for day,ids in byday.items():
        base=math.fsum((float(ens['BASE'][rid]['prediction'])-float(ens['BASE'][rid]['sub_ec']))**2 for rid in ids)
        arm=math.fsum((float(ens[a][rid]['prediction'])-float(ens[a][rid]['sub_ec']))**2 for rid in ids)
        vals.append(dict(farm=day[0],day=day[1],high=hi[day],benefit_sse=base-arm))
    for scope in ('all','normal','high'):
        v=[r for r in vals if scope=='all' or r['high']==(scope=='high')];pos=sorted((r for r in v if r['benefit_sse']>0),key=lambda r:r['benefit_sse'],reverse=True);neg=sorted((r for r in v if r['benefit_sse']<0),key=lambda r:r['benefit_sse'])
        grossgain=math.fsum(r['benefit_sse'] for r in pos);grossloss=-math.fsum(r['benefit_sse'] for r in neg)
        conc.append(dict(arm=a,scope=scope,days=len(v),better_days=len(pos),worse_days=len(neg),net_benefit_sse=math.fsum(r['benefit_sse'] for r in v),gross_gain_sse=grossgain,gross_loss_sse=grossloss,top5_gain_share=math.fsum(r['benefit_sse'] for r in pos[:5])/grossgain if grossgain else None,top5_loss_share=-math.fsum(r['benefit_sse'] for r in neg[:5])/grossloss if grossloss else None,largest5_gains=pos[:5],largest5_losses=neg[:5]))
out=dict(status='FINAL_SUPPLEMENT_PASS',alpha=alpha,seed_normal_better_than_base=seedbetter,discord_all_seeds_better_than_control=better_control,normal_screen=screen,utility_guard_trigger=guard,adoption=False,new_submission=False,new_candidate_fit=fit,new_baseline_fit=1,max_final_baseline_gap=maxgap,changes=changes,fold_normal_variation=foldvar,concentration=conc,receipts=receipts)
with (H/'final_critic_supplement_v1.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print('FIT',fit,'MAXBASEGAP',maxgap,'SCREEN',screen,'GUARD',guard)
print(json.dumps([r for r in changes if r['seed']=='ensemble' and r['scope'] in ('all','normal','high','pass2_normal','pass2_high','normal_without161')],ensure_ascii=False,indent=2))
print('CONCENTRATION',json.dumps(conc,ensure_ascii=False))
print('STRATA',json.dumps([r for r in changes if r['seed']=='ensemble' and ':' in r['scope']],ensure_ascii=False))
