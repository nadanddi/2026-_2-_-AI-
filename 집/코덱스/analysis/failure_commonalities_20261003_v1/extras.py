from patterns import *
def main():
 d=pd.read_csv(H/'final_daily_cases.csv');raw,t,e,frames=load();out=[]
 rr={k:v for k,v in RULES.items() if v[0] not in ['level_sse_fraction','all_under_fraction','all_over_fraction']}
 for tar,q in d.groupby('target'):
  for group in ['failure','core_failure']:
   f=q[group].to_numpy();strata=q.farm+'_'+q.late.astype(int).astype(str)
   for (n1,(c1,o1,v1)),(n2,(c2,o2,v2)) in itertools.combinations(rr.items(),2):
    valid=q[c1].notna().to_numpy()&q[c2].notna().to_numpy();a=condition(q[c1],o1,v1).to_numpy()&condition(q[c2],o2,v2).to_numpy();w=[];bg=[]
    for s in sorted(strata.unique()):
     yes=(strata==s).to_numpy()&f&valid;other=(strata==s).to_numpy()&~f&valid
     if yes.any() and other.any():w.append(yes.sum());bg.append(a[other].mean())
    rate=float(a[f&valid].mean()) if (f&valid).any() else np.nan;matched=float(np.average(bg,weights=w)) if w else np.nan
    good=q.good.to_numpy()&valid
    out.append(dict(target=tar,group=group,rule1=n1,rule2=n2,n=int((f&valid).sum()),yes=int(a[f&valid].sum()),rate=rate,matched_other_rate=matched,lift=rate-matched,good_n=int(good.sum()),good_yes=int(a[good].sum())))
 pd.DataFrame(out).to_csv(H/'pairwise_input_conditions.csv',index=False)
 stable=[]
 for tar,p in [('TEMP',t[t.member=='W30G']),('EC',e)]:
  p=p.copy();p['seed']=p.base_seed if tar=='TEMP' else p.seed;p['context']=p.context if tar=='TEMP' else '1-4';p['yy']=p.sub_temp if tar=='TEMP' else p.sub_ec;p['pp']=p.prediction if tar=='TEMP' else p.season_v2;p['se']=(p.pp-p.yy)**2
  for (seed,ctx),z in p.groupby(['seed','context']):
   g=z.groupby(['farm','day']).agg(y=('yy','mean'),pred=('pp','mean'),sse=('se','sum'),n=('row_id','size'));g['rmse']=np.sqrt(g.sse/g.n);g['bias']=g.pred-g.y
   for f,day in d[(d.target==tar)&d.failure][['farm','day']].itertuples(index=False,name=None):
    r=g.loc[(f,day)];stable.append(dict(target=tar,farm=f,day=int(day),seed=int(seed),context=ctx,rmse=float(r.rmse),bias=float(r.bias),remains_failure=bool(r.rmse>(.5 if tar=='TEMP' else .1)),remains_core=bool(abs(r.bias)>=.5 if tar=='TEMP' else r.bias<=-.2)))
 pd.DataFrame(stable).to_csv(H/'failure_stability.csv',index=False)
 # Adjacency reduces the number of independent episodes.
 episodes=[]
 for tar,q in d[d.failure].groupby('target'):
  for f,z in q.groupby('farm'):
   days=sorted(z.day);runs=[]
   for day in days:
    if not runs or day!=runs[-1][-1]+1:runs.append([int(day)])
    else:runs[-1].append(int(day))
   episodes.append(dict(target=tar,farm=f,failure_days=len(days),consecutive_episodes=len(runs),episodes=runs))
 save('episodes.json',episodes)
if __name__=='__main__':main()
