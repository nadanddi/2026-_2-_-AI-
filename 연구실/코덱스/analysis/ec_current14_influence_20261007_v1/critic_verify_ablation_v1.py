"""All deletion caches, pipeline and fsum scores; no model or tree refitting."""
from pathlib import Path
import sys,json,csv,math,collections
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import runner_v4 as R
import numpy as np,pandas as pd
raw,jobs,_=R.prepare();ARMS={'D1':[139],'D2':[139,231]}
receipt=json.loads((H/'ablation_receipt_v1.json').read_text(encoding='utf-8'));ap=H/'ablation_preparation_v1.json';reg=json.loads(ap.read_text(encoding='utf-8'))
assert receipt['source_sha']==reg['source_sha']==R.sha(H/'ablation_v1.py') and receipt['preparation_sha']==R.sha(ap)
assert reg['preparation_sha']==R.sha(H/'preparation_v4.json') and reg['baseline_receipt_sha']==R.sha(H/'baseline_receipt_v4.json') and reg['arms']==ARMS
assert receipt['rows_sha']==R.sha(R.L/'ablation_rows.csv')
read=lambda p:pd.read_csv(p,float_precision='round_trip',dtype={'seed':str})
rows=read(R.L/'ablation_rows.csv');base=read(R.L/'baseline_rows.csv')
assert len(rows)==69120 and not rows[['arm','seed','row_id']].duplicated().any()
assert set(rows.arm)==set(ARMS) and set(rows.seed)=={'7','101','2024','ensemble'}
worst={};cachechecks=[];interactions=[];status=[]
def close(name,a,b,tol=2e-12):
    a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name
    e=float(np.max(abs(a-b))) if a.size else 0.;worst[name]=max(worst.get(name,0.),e);assert e<=tol,(name,e)
metaindex={(m['arm'],m['k'],m['seed']):m for m in receipt['models']};assert len(metaindex)==60
for k,(t,q,pfn) in jobs.items():
    fixed={}
    for s in R.SEEDS:
        p=R.L/'base'/f'{k}_{s}.npz';m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert m['sha']==R.sha(p) and m['prep_sha']==R.sha(H/'preparation_v4.json')
        with np.load(p,allow_pickle=False) as c:
            assert c['train_row_id'].astype(str).tolist()==t.row_id.tolist() and c['row_id'].astype(str).tolist()==q.row_id.tolist();fixed[str(s)]={n:c[n].copy() for n in ['et','lgb','mlp']}
    ref=set(t[['farm','day']].itertuples(index=False,name=None));ec=t.groupby(['farm','day']).sub_ec.mean();S=R.SG.prepare(raw,ref);cal=R.SG.ref_calendar(S,ref);lo,hi=float(t.sub_ec.min()),float(t.sub_ec.max())
    lookup={}
    for i,(farm,day,hour) in enumerate(q[['farm','day','hour']].itertuples(index=False,name=None)):lookup.setdefault((farm,day),[]).append((hour,i))
    for arm,deleted in ARMS.items():
        removed=t.farm.eq('F47')&t.day.isin(deleted);tt=t.loc[~removed].reset_index(drop=True);et={}
        for s in R.SEEDS:
            p=R.L/'ablation'/f'{arm}_{k}_{s}.npz';m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert m==metaindex[(arm,k,s)]
            assert m['sha']==R.sha(p) and m['prep_sha']==R.sha(ap) and (m['arm'],m['k'],m['seed'])==(arm,k,s)
            h=R.sha.__module__ # no mutation; hash below rebuilt directly rather than runner.fhash
            import hashlib
            fh=hashlib.sha256(pd.util.hash_pandas_object(tt[['row_id','sub_ec']+R.M.FULL_R3],index=False).to_numpy().tobytes()).hexdigest();assert m['train_hash']==fh
            assert m['removed_rows']==int(removed.sum()) and m['new_fit']==bool(removed.any())
            with np.load(p,allow_pickle=False) as c:
                assert c['row_id'].astype(str).tolist()==q.row_id.tolist() and c['train_row_id'].astype(str).tolist()==tt.row_id.tolist()
                assert c['removed_row_id'].astype(str).tolist()==t.loc[removed,'row_id'].tolist()
                assert c['et'].shape==(len(q),) and np.isfinite(c['et']).all();et[str(s)]=c['et'].copy()
            if not removed.any():assert np.array_equal(et[str(s)],fixed[str(s)]['et'])
            cachechecks.append(dict(arm=arm,fold=k,seed=s,removed_rows=int(removed.sum()),new_fit=bool(removed.any()),sha=R.sha(p)))
        et['ensemble']=np.array([math.fsum(v)/3 for v in zip(*[et[str(s)] for s in R.SEEDS])])
        for tag in ['7','101','2024','ensemble']:
            r=rows[(rows.k==k)&rows.arm.eq(arm)&rows.seed.eq(tag)].set_index('row_id').loc[q.row_id].reset_index();b=base[(base.k==k)&base.seed.eq(tag)].set_index('row_id').loc[q.row_id].reset_index()
            assert len(r)==len(q);close('labels',r.sub_ec,q.sub_ec);close('raw_et',r.raw_et,et[tag]);assert r.clip_lo.eq(lo).all() and r.clip_hi.eq(hi).all()
            lg=fixed[tag]['lgb'] if tag!='ensemble' else np.mean([fixed[str(s)]['lgb'] for s in R.SEEDS],axis=0)
            mlp=fixed[tag]['mlp'] if tag!='ensemble' else np.mean([fixed[str(s)]['mlp'] for s in R.SEEDS],axis=0)
            mix=np.array([math.fsum([.48*a,.24*c,.08*d,.2*p]) for a,c,d,p in zip(et[tag],lg,mlp,pfn)]);close('fixed_other_members_raw_mix',mix,r.raw_mix)
            smooth=np.empty(len(q))
            for g in lookup.values():
                prefix=[]
                for hour,i in sorted(g):prefix.append(mix[i]);smooth[i]=.5*mix[i]+.5*math.fsum(prefix)/len(prefix)
            close('smooth',smooth,r.smooth);pre=np.clip(smooth,lo,hi);close('pre_sg2',pre,r.pre_sg2)
            corr=R.M.sg2post.correct(q[['row_id']],pre,S,ec,ref,cal);close('original_sg2',corr,r.sg2_raw);close('final',np.clip(corr,lo,hi),r.prediction)
            # Candidate identity/calendar is entirely input/ref-driven and must stay fixed.
            for col in ['candidate_day','candidate_ec','query_calendar']:np.testing.assert_allclose(r[col],b[col],rtol=0,atol=0,equal_nan=True)
            present=r.candidate_day.notna().to_numpy();p2=q.day.ge(179).to_numpy();gate=r.gate.to_numpy(bool);assert not np.any(present&~p2) and not np.any(gate&~present)
            for g in lookup.values():
                prefix=[]
                for hour,i in sorted(g):
                    prefix.append(pre[i]);v=r.iloc[i]
                    if not present[i]:continue
                    pm=math.fsum(prefix)/len(prefix);close('prefix_model',[pm],[v.prefix_model]);a1=float(ec[(v.farm,int(v.candidate_day))]);close('candidate_label',[a1],[v.candidate_ec]);allowed=abs(a1-pm)<=.30;assert bool(v.gate)==allowed;close('delta',[.5*(a1-pm) if allowed else 0.],[v.delta])
            status.append(dict(arm=arm,seed=tag,fold=k,pass1_inactive=int((~p2).sum()),pass2_no_candidate=int((p2&~present).sum()),pass2_gate_false=int((p2&present&~gate).sum()),pass2_gate_true=int((p2&gate).sum()),preclip_rows=int((abs(smooth-pre)>1e-12).sum()),postclip_rows=int((abs(corr-np.clip(corr,lo,hi))>1e-12).sum())))
assert receipt['new_ET_fits']==sum(c['new_fit'] for c in cachechecks) and receipt['additional_baseline_trace_fit']==1
# Lightweight trace sidecar linkage only. No duplicate 1800-tree route audit.
tracechecks=[]
for arm in ['BASE','D1','D2']:
    p=R.L/'trace'/f'{arm}.npz';m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert m['sha']==R.sha(p) and (m['arm'],m['seed'],m['fold'])==(arm,7,1) and m['source_sha']==R.sha(H/'ablation_v1.py')
    cp=R.L/('base/1_7.npz' if arm=='BASE' else f'ablation/{arm}_1_7.npz');cm=json.loads(cp.with_suffix('.json').read_text(encoding='utf-8'));assert cm['sha']==R.sha(cp) and cm['prep_sha']==R.sha(H/'preparation_v4.json' if arm=='BASE' else ap)
    with np.load(p,allow_pickle=False) as z,np.load(cp,allow_pickle=False) as c:
        assert z['train_row_id'].tolist()==c['train_row_id'].tolist();ix=pd.Index(c['row_id']).get_indexer(z['query_row_id']);assert (ix>=0).all();close('trace_cached_raw_et',z['raw_et'],c['et'][ix],1e-10);assert int(z['offsets'][-1])==m['nodes'] and len(z['offsets'])==601
    tracechecks.append(dict(arm=arm,seed=7,fold=1,sha=R.sha(p),nodes=m['nodes'],source_sha=m['source_sha']))
# csv/fsum daily scoring; BASE/OLD exact score-copy validation plus new D1/D2 arithmetic.
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
state={}
for (farm,day),g in R.M.identify(raw).groupby(['farm','day']):
    fan=[float(x) for x in g.act_circfan if pd.notna(x)];state[(farm,int(day))]=(sum(v==0 for v in g.act_vent)/len(g),math.fsum(fan)/len(fan))
daily={};grouped=collections.defaultdict(list)
for r in csvread(R.L/'ablation_rows.csv'):grouped[(r['arm'],r['seed'],r['farm'],int(r['day']))].append(r)
for key,g in grouped.items():
    truth=math.fsum(float(r['sub_ec']) for r in g)/24;pred=[float(r['prediction']) for r in g];errors=[p-float(r['sub_ec']) for p,r in zip(pred,g)];sse=math.fsum(x*x for x in errors);bias=math.fsum(errors)/24
    daily[key]=dict(truth=truth,prediction=math.fsum(pred)/24,bias=bias,sse=sse,rmse=math.sqrt(sse/24),sse_day=24*bias*bias,sse_shape=sse-24*bias*bias,raw_et=math.fsum(float(r['raw_et']) for r in g)/24,k=int(g[0]['k']))
D=H/'ablation_score_v1';comp=json.loads((D/'completion.json').read_text(encoding='utf-8'));assert comp['input_sha']==R.sha(R.L/'ablation_rows.csv') and comp['source_sha']==R.sha(H/'score_v1.py')
for name,h in comp['files'].items():assert R.sha(D/name)==h
baseline_days={(r['arm'],r['seed'],r['farm'],int(r['day'])):r for r in csvread(H/'baseline_score_v1/days.csv')}
scored=csvread(D/'days.csv');assert len(scored)==5760
for r in scored:
    key=(r['arm'],r['seed'],r['farm'],int(r['day']))
    if r['arm'] in ['BASE','OLD_SEASON']:
        assert r==baseline_days[key]
        br=baseline_days[key];daily[key]={n:float(br[n]) if br[n] else None for n in ['truth','prediction','bias','sse','rmse','sse_day','sse_shape','raw_et']};daily[key]['k']=int(br['k'])
    else:
        d=daily[key]
        for n in ['truth','prediction','bias','sse','rmse','sse_day','sse_shape','raw_et']:close('daily_'+n,[float(r[n])],[d[n]],2e-11)
def included(group,key,d):
    arm,seed,farm,day=key;ordinary=d['truth']<1;closed=state[(farm,day)][0]>=.8;fanlow=state[(farm,day)][1]<10
    return {'all':True,'ordinary':ordinary,'high':not ordinary,'ordinary_closed61':ordinary and closed,'ordinary_other268':ordinary and not closed,'ordinary_closed_fanlow50':ordinary and closed and fanlow,'pass1':day<179,'pass2_public46':day>=179,'F13':farm=='F13','F47':farm=='F47','excluding_F47_161':(farm,day)!=('F47',161),'ordinary_excluding_F47_161':ordinary and (farm,day)!=('F47',161)}[group]
stats=csvread(D/'groups.csv');assert len(stats)==192
for r in stats:
    d=[v for k,v in daily.items() if k[0]==r['arm'] and k[1]==r['seed'] and included(r['group'],k,v)];sse=math.fsum(v['sse'] for v in d);rmse=math.sqrt(sse/(24*len(d)));assert int(r['days'])==len(d) and int(r['rows'])==24*len(d)
    close('group_sse',[sse],[float(r['sse'])],2e-10);close('group_rmse',[rmse],[float(r['rmse'])]);close('group_bias',[math.fsum(v['bias'] for v in d)/len(d)],[float(r['bias'])]);assert int(r['good_days'])==sum(v['rmse']<=.1 for v in d) and int(r['severe_days'])==sum(v['rmse']>=.2 for v in d)
    close('day_fraction',[math.fsum(v['sse_day'] for v in d)/sse],[float(r['day_level_sse_fraction'])]);b=[v for k,v in daily.items() if k[0]=='BASE' and k[1]==r['seed'] and included(r['group'],k,v)];bsse=math.fsum(v['sse'] for v in b);brmse=math.sqrt(bsse/(24*len(b)));close('base_sse',[bsse],[float(r['base_sse'])],2e-10);close('base_rmse',[brmse],[float(r['base_rmse'])]);close('delta_sse',[sse-bsse],[float(r['delta_sse'])],2e-10);close('rmse_percent',[100*(rmse/brmse-1)],[float(r['rmse_percent'])],2e-10)
cases=csvread(D/'cases.csv');expected={k for k in daily if (k[2],k[3]) in {('F47',160),('F47',161),('F13',98),('F13',112)}};assert len(cases)==64 and {(r['arm'],r['seed'],r['farm'],int(r['day'])) for r in cases}==expected
scoredlookup={(r['arm'],r['seed'],r['farm'],int(r['day'])):r for r in scored}
for r in cases:assert r==scoredlookup[(r['arm'],r['seed'],r['farm'],int(r['day']))]
tags={}
for arm in ARMS:
    b=[daily[('BASE',str(s),'F47',161)] for s in R.SEEDS];a=[daily[(arm,str(s),'F47',161)] for s in R.SEEDS];bias=[v['raw_et']-v['truth'] for v in a];tags[arm]=dict(SELECTED_CASE_DEPENDENCE=all(abs(x)<abs(v['raw_et']-v['truth']) for x,v in zip(bias,b)),RESIDUAL_STATE_CONFUSION=all(x>.2 for x in bias),raw_et_day_bias=dict(zip(map(str,R.SEEDS),bias)))
    assert comp['tags'][arm]['SELECTED_CASE_DEPENDENCE']==tags[arm]['SELECTED_CASE_DEPENDENCE'] and comp['tags'][arm]['RESIDUAL_STATE_CONFUSION']==tags[arm]['RESIDUAL_STATE_CONFUSION']
    for s in R.SEEDS:close('tag_raw_bias',[comp['tags'][arm]['raw_et_day_bias'][str(s)]],[tags[arm]['raw_et_day_bias'][str(s)]])
for arm in ['BASE',*ARMS]:
    a=base if arm=='BASE' else rows[rows.arm.eq(arm)];actual=a[a.seed.eq('ensemble')].set_index('row_id').prediction;mean=a[a.seed.ne('ensemble')].groupby('row_id').prediction.mean();diff=actual-mean;interactions.append(dict(kind='actual_ensemble_vs_mean_post',arm=arm,changed_rows=int((abs(diff)>1e-12).sum()),maxdiff=float(abs(diff).max()),mean_absdiff=float(abs(diff).mean())))
    if arm=='BASE':continue
    for tag in ['7','101','2024','ensemble']:
        b=base[base.seed.eq(tag)].set_index('row_id');v=a[a.seed.eq(tag)].set_index('row_id').loc[b.index];baseline_et=np.concatenate([fixed[str(R.SEEDS[0])]['et'][:0]]) # placeholder unused
        # BASE rawET is independently stored in already-verified baseline day rows, not output CSV.
        etmap={}
        for k,(t,q,pfn) in jobs.items():
            arr=[]
            for s in R.SEEDS:
                with np.load(R.L/'base'/f'{k}_{s}.npz',allow_pickle=False) as c:arr.append(c['et'].copy())
            z=arr[R.SEEDS.index(int(tag))] if tag!='ensemble' else np.mean(arr,axis=0)
            etmap.update(zip(q.row_id,z))
        be=np.array([etmap[i] for i in b.index]);interactions.append(dict(kind='ET_change_and_SG2_interaction',arm=arm,seed=tag,changed_et_rows=int((abs(v.raw_et.to_numpy()-be)>1e-12).sum()),changed_final_rows=int((abs(v.prediction-b.prediction)>1e-12).sum()),gate_flips=int((v.gate!=b.gate).sum()),reference_candidate_changed=int(((v.candidate_day.fillna(-1)!=b.candidate_day.fillna(-1))|(v.candidate_ec.fillna(-1)!=b.candidate_ec.fillna(-1))).sum()),max_direct_pre_sg2_change=float(abs(v.pre_sg2-b.pre_sg2).max()),max_sg2_interaction=float(abs((v.sg2_raw-v.pre_sg2)-(b.sg2_raw-b.pre_sg2)).max())))
saved=csvread(D/'interactions.csv');assert len(saved)==len(interactions)
for r,e in zip(saved,interactions):
    for n,v in e.items():
        if isinstance(v,str):assert r[n]==v
        else:close('interaction_'+n,[float(r[n])],[v])
totals={f'{a}_{s}':{n:sum(v[n] for v in status if v['arm']==a and v['seed']==s) for n in status[0] if n not in ['fold','arm','seed']} for a in ARMS for s in ['7','101','2024','ensemble']}
out=dict(status='PASS_ABLATION_CACHE_PIPELINE_FSUM_SCORES_AND_FIXED_TAGS',fit=0,caches=60,rows=69120,new_deletion_fits=sum(c['new_fit'] for c in cachechecks),no_op_caches=sum(not c['new_fit'] for c in cachechecks),baseline_replay_fits=1,day_scores=5760,group_scores=192,cases=64,max_errors=worst,tags=tags,SG2_status=totals,interactions=interactions,trace_sidecars=tracechecks,cache_checks=cachechecks,scope='Independent cache/outputs/scores audit; no duplicate fulltree routing. Parent support audit code and receipt reviewed separately.')
with (H/'critic_verify_ablation_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(out['status'],out['tags'],flush=True)
