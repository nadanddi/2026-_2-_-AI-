"""Complete baseline audit: no fitting; fsum scoring and stage reconstruction."""
from pathlib import Path
import sys,json,csv,math,collections
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import runner_v4 as R
import numpy as np,pandas as pd
raw,jobs,prep=R.prepare()
receipt=json.loads((H/'baseline_receipt_v4.json').read_text(encoding='utf-8'))
p=R.L/'baseline_rows.csv';assert R.sha(p)==receipt['rows_sha'] and R.sha(H/'preparation_v4.json')==receipt['prep_sha']
rows=pd.read_csv(p,float_precision='round_trip',dtype={'seed':str})
assert len(rows)==34560 and not rows[['seed','row_id']].duplicated().any()
assert set(rows.seed)=={'7','101','2024','ensemble'} and set(rows.arm)=={'BASE'}
worst={};status=[];oldpred={};baseet={};ensemble=[]
def close(name,a,b,tol=2e-12):
    a=np.asarray(a,float);b=np.asarray(b,float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name
    e=float(np.max(abs(a-b))) if a.size else 0.;worst[name]=max(worst.get(name,0.),e);assert e<=tol,(name,e)
for k,(t,q,pfn) in jobs.items():
    member={};oldmix=[]
    for seed in R.SEEDS:
        dest=R.L/'base'/f'{k}_{seed}.npz';m=json.loads(dest.with_suffix('.json').read_text(encoding='utf-8'))
        assert m['sha']==R.sha(dest) and m['prep_sha']==R.sha(H/'preparation_v4.json') and (m['k'],m['seed'])==(k,seed)
        with np.load(dest,allow_pickle=False) as c:
            assert c['row_id'].astype(str).tolist()==q.row_id.tolist() and c['train_row_id'].astype(str).tolist()==t.row_id.tolist()
            member[str(seed)]={n:c[n].copy() for n in ['et','lgb','mlp']}
            for n in ['et','lgb','mlp']:assert c[n].shape==(len(q),) and np.isfinite(c[n]).all()
        with np.load(R.C/f'DIAG10_{k}_r3_{seed}.npz',allow_pickle=False) as c:
            assert c['row_id'].astype(str).tolist()==q.row_id.tolist()
            oldmix.append(np.array([math.fsum([.48*a,.24*b,.08*c1,.2*d]) for a,b,c1,d in zip(c['raw_et'],c['raw_lgb'],c['raw_mlp'],pfn)]))
    member['ensemble']={n:np.mean([member[str(s)][n] for s in R.SEEDS],axis=0) for n in ['et','lgb','mlp']}
    ref=set(t[['farm','day']].itertuples(index=False,name=None));ec=t.groupby(['farm','day']).sub_ec.mean();S=R.SG.prepare(raw,ref);cal=R.SG.ref_calendar(S,ref)
    lo,hi=float(t.sub_ec.min()),float(t.sub_ec.max())
    for j,tag in enumerate(['7','101','2024','ensemble']):
        r=rows[(rows.k==k)&(rows.seed==tag)].set_index('row_id').loc[q.row_id].reset_index();assert len(r)==len(q)
        close('labels',r.sub_ec,q.sub_ec);assert r.clip_lo.eq(lo).all() and r.clip_hi.eq(hi).all()
        a=member[tag];mix=np.array([math.fsum([.48*v,.24*w,.08*x,.2*y]) for v,w,x,y in zip(a['et'],a['lgb'],a['mlp'],pfn)])
        close('raw_mix',mix,r.raw_mix)
        smooth=np.empty(len(q));lookup={}
        for i,(farm,day,hour) in enumerate(q[['farm','day','hour']].itertuples(index=False,name=None)):lookup.setdefault((farm,day),[]).append((hour,i))
        for group in lookup.values():
            prefix=[]
            for hour,i in sorted(group):prefix.append(mix[i]);smooth[i]=math.fsum([.5*mix[i],.5*math.fsum(prefix)/len(prefix)])
        close('smooth',smooth,r.smooth);pre=np.clip(smooth,lo,hi);close('pre_sg2',pre,r.pre_sg2)
        corr=R.M.sg2post.correct(q[['row_id']],pre,S,ec,ref,cal)
        close('sg2_raw_original_function',corr,r.sg2_raw);close('final',np.clip(corr,lo,hi),r.prediction)
        # Saved traces are independently checked against reference labels and base-prefix means.
        trace_present=r.candidate_day.notna().to_numpy();p2=q.day.ge(179).to_numpy();gate=r.gate.to_numpy(bool)
        assert not np.any(trace_present&~p2) and not np.any(gate&~trace_present)
        for indices in lookup.values():
            prefix=[]
            for hour,i in sorted(indices):
                prefix.append(pre[i]);v=r.iloc[i]
                if not trace_present[i]:continue
                key=(str(v.farm),int(v.candidate_day));assert key in ref
                close('candidate_ec',[v.candidate_ec],[ec[key]])
                pm=math.fsum(prefix)/len(prefix);close('prefix_model',[pm],[v.prefix_model])
                allowed=abs(float(ec[key])-pm)<=.30;assert bool(v.gate)==allowed
                close('trace_delta',[v.delta],[.5*(float(ec[key])-pm) if allowed else 0.])
                # Independent signature/cal-distance ranking over the saved eligible calendar.
                eligible=[d for f,d in sorted(ref) if f==v.farm and abs(cal[(f,d)]-v.query_calendar)<=3 and cal[(f,d)]!=v.query_calendar]
                assert eligible
                rs=S['SIG'][int(hour)].loc[[(v.farm,d) for f,d in sorted(ref) if f==v.farm]].astype(float)
                mu=rs.mean().to_numpy();sd=rs.std().replace(0,np.nan).to_numpy();query=(S['SIG'][int(hour)].loc[(v.farm,int(v.day))].to_numpy(float)-mu)/sd;use=~np.isnan(query)
                c=np.nan_to_num(((S['SIG'][int(hour)].loc[[(v.farm,d) for d in eligible]].to_numpy(float)-mu)/sd)[:,use])
                distances=np.sqrt(((c-query[use])**2).mean(axis=1))+.15*np.array([abs(cal[(v.farm,d)]-v.query_calendar) for d in eligible])
                assert eligible[int(np.argmin(distances))]==int(v.candidate_day)
        status.append(dict(fold=k,seed=tag,pass1_inactive=int((~p2).sum()),pass2_no_candidate=int((p2&~trace_present).sum()),pass2_gate_false=int((p2&trace_present&~gate).sum()),pass2_gate_true=int((p2&gate).sum()),nonzero_delta=int((r.delta.fillna(0).abs()>1e-12).sum()),preclip_rows=int((abs(smooth-pre)>1e-12).sum()),postclip_rows=int((abs(corr-np.clip(corr,lo,hi))>1e-12).sum())))
        for i,rid in enumerate(q.row_id):baseet[(tag,rid)]=float(a['et'][i])
        om=oldmix[j] if j<3 else np.mean(oldmix,axis=0)
        op=np.empty(len(q))
        for group in lookup.values():
            prefix=[]
            for h,i in sorted(group):prefix.append(om[i]);op[i]=min(hi,max(lo,.5*om[i]+.5*math.fsum(prefix)/len(prefix)))
        for i,rid in enumerate(q.row_id):oldpred[(tag,rid)]=float(op[i])
# Independent csv/math.fsum score table verification.
with p.open(encoding='utf-8',newline='') as f:original=list(csv.DictReader(f))
state={}
for (farm,day),g in R.M.identify(raw).groupby(['farm','day']):
    fan=[float(v) for v in g.act_circfan if pd.notna(v)];state[(farm,int(day))]=(sum(v==0 for v in g.act_vent)/len(g),math.fsum(fan)/len(fan))
daily={}
for arm in ['BASE','OLD_SEASON']:
    groups=collections.defaultdict(list)
    for r in original:groups[(r['seed'],r['farm'],int(r['day']))].append(r)
    for key,g in groups.items():
        truth=math.fsum(float(r['sub_ec']) for r in g)/24;pred=[float(r['prediction']) if arm=='BASE' else oldpred[(r['seed'],r['row_id'])] for r in g];errors=[v-float(r['sub_ec']) for v,r in zip(pred,g)];sse=math.fsum(e*e for e in errors);bias=math.fsum(errors)/24
        daily[(arm,*key)]=dict(n=24,truth=truth,prediction=math.fsum(pred)/24,bias=bias,sse=sse,rmse=math.sqrt(sse/24),sse_day=24*bias*bias,sse_shape=sse-24*bias*bias,raw_et=math.fsum(baseet[(key[0],r['row_id'])] for r in g)/24 if arm=='BASE' else None,k=int(g[0]['k']))
score=H/'baseline_score_v1';completion=json.loads((score/'completion.json').read_text(encoding='utf-8'));assert completion['input_sha']==R.sha(p)
for name,hash1 in completion['files'].items():assert R.sha(score/name)==hash1
with (score/'days.csv').open(encoding='utf-8',newline='') as f:
    scored=list(csv.DictReader(f))
assert len(scored)==2880
for r in scored:
    d=daily[(r['arm'],r['seed'],r['farm'],int(r['day']))]
    for n in ['truth','prediction','bias','sse','rmse','sse_day','sse_shape']:close('daily_'+n,[float(r[n])],[d[n]],2e-11)
    if r['arm']=='BASE':close('daily_raw_et',[float(r['raw_et'])],[d['raw_et']])
def included(group,key,d):
    arm,seed,farm,day=key;ordinary=d['truth']<1;closed=state[(farm,day)][0]>=.8;fanlow=state[(farm,day)][1]<10
    return {'all':True,'ordinary':ordinary,'high':not ordinary,'ordinary_closed61':ordinary and closed,'ordinary_other268':ordinary and not closed,'ordinary_closed_fanlow50':ordinary and closed and fanlow,'pass1':day<179,'pass2_public46':day>=179,'F13':farm=='F13','F47':farm=='F47','excluding_F47_161':(farm,day)!=('F47',161),'ordinary_excluding_F47_161':ordinary and (farm,day)!=('F47',161)}[group]
with (score/'groups.csv').open(encoding='utf-8',newline='') as f:scoredgroups=list(csv.DictReader(f))
for r in scoredgroups:
    d=[v for k,v in daily.items() if k[0]==r['arm'] and k[1]==r['seed'] and included(r['group'],k,v)];sse=math.fsum(v['sse'] for v in d);rmse=math.sqrt(sse/(24*len(d)));assert int(r['days'])==len(d) and int(r['rows'])==24*len(d)
    close('group_sse',[sse],[float(r['sse'])],2e-10);close('group_rmse',[rmse],[float(r['rmse'])]);close('group_bias',[math.fsum(v['bias'] for v in d)/len(d)],[float(r['bias'])]);assert int(r['good_days'])==sum(v['rmse']<=.1 for v in d) and int(r['severe_days'])==sum(v['rmse']>=.2 for v in d)
    close('day_fraction',[math.fsum(v['sse_day'] for v in d)/sse],[float(r['day_level_sse_fraction'])])
    b=[v for k,v in daily.items() if k[0]=='BASE' and k[1]==r['seed'] and included(r['group'],k,v)];bsse=math.fsum(v['sse'] for v in b);brmse=math.sqrt(bsse/(24*len(b)))
    close('delta_sse',[sse-bsse],[float(r['delta_sse'])],2e-10);close('rmse_percent',[100*(rmse/brmse-1)],[float(r['rmse_percent'])],2e-10)
actual=rows[rows.seed=='ensemble'].set_index('row_id').prediction;mean=rows[rows.seed!='ensemble'].groupby('row_id').prediction.mean();diff=actual-mean
interaction=dict(changed_rows=int((abs(diff)>1e-12).sum()),maxdiff=float(abs(diff).max()),mean_absdiff=float(abs(diff).mean()))
saved=pd.read_csv(score/'interactions.csv').iloc[0];assert int(saved.changed_rows)==interaction['changed_rows'];close('ensemble_maxdiff',[saved.maxdiff],[interaction['maxdiff']]);close('ensemble_mean_absdiff',[saved.mean_absdiff],[interaction['mean_absdiff']])
totals={tag:{n:sum(s[n] for s in status if s['seed']==tag) for n in status[0] if n not in ['fold','seed']} for tag in ['7','101','2024','ensemble']}
out=dict(status='PASS_BASELINE_CACHE_PIPELINE_AND_FSUM_SCORES',fit=0,caches=30,rows=len(rows),days=360,score_daily_rows=len(scored),group_rows=len(scoredgroups),max_errors=worst,SG2_status=totals,ensemble_vs_mean_post=interaction,limitation='No model refit. Current immutable features and original SG2 correct reused; saved candidate ranking/delta independently recomputed. Physical causality/adoption not tested.')
with (H/'critic_verify_baseline_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False),flush=True)
