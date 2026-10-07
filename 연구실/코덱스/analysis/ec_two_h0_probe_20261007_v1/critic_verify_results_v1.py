"""Full cache/pipeline/fsum and bootstrap replay audit. No fitting/tree routing."""
from pathlib import Path
import sys,json,csv,math,collections,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as R
np,pd,B=R.np,R.pd,R.B
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
raw,jobs,prep=R.prepare();receipt=json.loads((H/'receipt_v1.json').read_text(encoding='utf-8'));assert receipt['rows_sha']==sha(R.L/'rows.csv') and receipt['source_sha']==sha(H/'run_v1.py') and receipt['preparation_sha']==sha(H/'preparation_v1.json') and receipt['new_ET_fits']==30 and receipt['other_new_fit']==0
manifest=json.loads((H/'implementation_manifest_v1.json').read_text(encoding='utf-8'))
for name,value in manifest['files'].items():assert sha(H/name if (H/name).exists() else R.ROOT/name)==value
rows=pd.read_csv(R.L/'rows.csv',float_precision='round_trip',dtype={'seed':str});oldrows=pd.read_csv(B.L/'baseline_rows.csv',float_precision='round_trip',dtype={'seed':str});assert len(rows)==69120 and not rows[['arm','seed','row_id']].duplicated().any() and set(rows.arm)=={'BASE',R.ARM} and set(rows.seed)=={'7','101','2024','ensemble'}
assert not (R.L/'worker.lock').exists();maxerr={};cachechecks=[]
def close(name,a,b,tol=2e-12):
    a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name
    e=float(np.max(abs(a-b))) if a.size else 0.;maxerr[name]=max(maxerr.get(name,0.),e);assert e<=tol,(name,e)
metas={(m['k'],m['seed']):m for m in receipt['models']};assert len(metas)==30
for k,(t,q,pfn) in jobs.items():
    rec=next(x for x in prep['records'] if x['k']==k)
    for kind,frame in [('train',t),('query',q)]:assert hashlib.sha256(pd.util.hash_pandas_object(frame[['row_id','sub_ec']+R.COLS],index=False).to_numpy().tobytes()).hexdigest()==rec[kind+'_hash']
    keep=[B.M.FULL_R3.index(c) for c in R.COLS];assert np.array_equal(np.nanmedian(t[R.COLS],axis=0),np.nanmedian(t[B.M.FULL_R3],axis=0)[keep])
    fixed={};ets={}
    for seed in B.SEEDS:
        p=B.L/'base'/f'{k}_{seed}.npz';m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert m['sha']==sha(p)==rec['base_caches'][str(seed)] and m['prep_sha']==prep['old_preparation_sha'] and (m['k'],m['seed'])==(k,seed)
        with np.load(p,allow_pickle=False) as z:
            assert z['train_row_id'].tolist()==t.row_id.tolist() and z['row_id'].tolist()==q.row_id.tolist();fixed[str(seed)]={n:z[n].copy() for n in ['et','lgb','mlp']}
            for n in fixed[str(seed)]:assert z[n].shape==(len(q),) and np.isfinite(z[n]).all()
        p=R.L/f'{k}_{seed}.npz';m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert m==metas[(k,seed)] and m['sha']==sha(p) and m['prep_sha']==sha(H/'preparation_v1.json') and m['source_sha']==prep['source_sha'] and (m['k'],m['seed'])==(k,seed) and m['train_hash']==rec['train_hash'] and m['query_hash']==rec['query_hash']
        with np.load(p,allow_pickle=False) as z:
            assert z['train_row_id'].tolist()==t.row_id.tolist() and z['row_id'].tolist()==q.row_id.tolist() and z['et'].shape==(len(q),) and np.isfinite(z['et']).all();ets[str(seed)]=z['et'].copy()
        cachechecks.append(dict(fold=k,seed=seed,sha=sha(p)))
    ets['ensemble']=np.array([math.fsum(v)/3 for v in zip(*[ets[str(s)] for s in B.SEEDS])])
    ref=set(t[['farm','day']].itertuples(index=False,name=None));ec=t.groupby(['farm','day']).sub_ec.mean();S=B.SG.prepare(raw,ref);cal=B.SG.ref_calendar(S,ref);lo,hi=float(t.sub_ec.min()),float(t.sub_ec.max());lookup={}
    for i,(f,d,h) in enumerate(q[['farm','day','hour']].itertuples(index=False,name=None)):lookup.setdefault((f,d),[]).append((h,i))
    for tag in ['7','101','2024','ensemble']:
        a=rows[rows.arm.eq(R.ARM)&rows.seed.eq(tag)&rows.k.eq(k)].set_index('row_id').loc[q.row_id].reset_index();b=rows[rows.arm.eq('BASE')&rows.seed.eq(tag)&rows.k.eq(k)].set_index('row_id').loc[q.row_id].reset_index();ob=oldrows[oldrows.seed.eq(tag)&oldrows.k.eq(k)].set_index('row_id').loc[q.row_id].reset_index();assert len(a)==len(b)==len(q)
        for col in ob:
            if col in ['arm','seed','row_id']:assert b[col].tolist()==ob[col].tolist()
            else:pd.testing.assert_series_equal(b[col],ob[col],check_names=False)
        be=fixed[tag]['et'] if tag!='ensemble' else np.mean([fixed[str(s)]['et'] for s in B.SEEDS],axis=0);close('BASE_raw_et',b.raw_et,be);close('alt_raw_et',a.raw_et,ets[tag]);close('labels',a.sub_ec,q.sub_ec)
        assert a.clip_lo.eq(lo).all() and a.clip_hi.eq(hi).all()
        lg=fixed[tag]['lgb'] if tag!='ensemble' else np.mean([fixed[str(s)]['lgb'] for s in B.SEEDS],axis=0);mlp=fixed[tag]['mlp'] if tag!='ensemble' else np.mean([fixed[str(s)]['mlp'] for s in B.SEEDS],axis=0)
        mix=np.array([math.fsum([.48*x,.24*y,.08*z,.2*p]) for x,y,z,p in zip(ets[tag],lg,mlp,pfn)]);close('fixed_other_raw_mix',mix,a.raw_mix);smooth=np.empty(len(q))
        for g in lookup.values():
            prefix=[]
            for h,i in sorted(g):prefix.append(mix[i]);smooth[i]=.5*mix[i]+.5*math.fsum(prefix)/len(prefix)
        close('prefix_smooth',smooth,a.smooth);pre=np.clip(smooth,lo,hi);close('pre_sg2',pre,a.pre_sg2);corr=B.M.sg2post.correct(q[['row_id']],pre,S,ec,ref,cal);close('original_sg2',corr,a.sg2_raw);close('final',np.clip(corr,lo,hi),a.prediction)
        for col in ['query_calendar','candidate_day','candidate_ec']:np.testing.assert_allclose(a[col],b[col],rtol=0,atol=0,equal_nan=True)
        for g in lookup.values():
            prefix=[]
            for h,i in sorted(g):
                prefix.append(pre[i]);v=a.iloc[i]
                if pd.isna(v.candidate_day):assert not v.gate;continue
                a1=float(ec[(v.farm,int(v.candidate_day))]);pm=math.fsum(prefix)/len(prefix);close('candidate_ec',[v.candidate_ec],[a1]);close('prefix_model',[v.prefix_model],[pm]);allowed=abs(a1-pm)<=.30;assert bool(v.gate)==allowed;close('delta',[v.delta],[.5*(a1-pm) if allowed else 0.])
# Independent all-row csv/fsum score arithmetic.
grouped=collections.defaultdict(list)
for r in read(R.L/'rows.csv'):grouped[(r['arm'],r['seed'],r['farm'],int(r['day']))].append(r)
daily={}
for key,g in grouped.items():
    assert len(g)==24;truth=math.fsum(float(r['sub_ec']) for r in g)/24;pred=[float(r['prediction']) for r in g];err=[p-float(r['sub_ec']) for p,r in zip(pred,g)];sse=math.fsum(e*e for e in err);bias=math.fsum(err)/24
    daily[key]=dict(truth=truth,prediction=math.fsum(pred)/24,bias=bias,sse=sse,rmse=math.sqrt(sse/24),sse_day=24*bias*bias,sse_shape=sse-24*bias*bias,raw_et=math.fsum(float(r['raw_et']) for r in g)/24)
D=H/'score_v1';comp=json.loads((D/'completion.json').read_text(encoding='utf-8'));assert comp['input_sha']==sha(R.L/'rows.csv') and comp['source_sha']==sha(H/'score_v1.py')
for name,value in comp['files'].items():assert sha(D/name)==value
scored=read(D/'days.csv');assert len(scored)==2880
for r in scored:
    key=(r['arm'],r['seed'],r['farm'],int(r['day']))
    for n,v in daily[key].items():close('daily_'+n,[r[n]],[v],2e-11)
state={}
for (f,d),g in B.M.identify(raw).groupby(['farm','day']):
    fan=[float(v) for v in g.act_circfan if pd.notna(v)];state[(f,int(d))]=(sum(v==0 for v in g.act_vent)/len(g),math.fsum(fan)/len(fan))
def included(group,key,d):
    arm,seed,farm,day=key;o=d['truth']<1;c=state[(farm,day)][0]>=.8;fl=state[(farm,day)][1]<10
    return {'all':True,'ordinary':o,'high':not o,'ordinary_closed61':o and c,'ordinary_other268':o and not c,'ordinary_closed_fanlow50':o and c and fl,'pass1':day<179,'pass2_public46':day>=179,'F13':farm=='F13','F47':farm=='F47','excluding_F47_161':(farm,day)!=('F47',161),'ordinary_excluding_F47_161':o and (farm,day)!=('F47',161)}[group]
stats=read(D/'groups.csv');assert len(stats)==96
for r in stats:
    selected=[v for k,v in daily.items() if k[0]==r['arm'] and k[1]==r['seed'] and included(r['group'],k,v)];sse=math.fsum(v['sse'] for v in selected);rmse=math.sqrt(sse/(24*len(selected)));assert int(r['days'])==len(selected) and int(r['rows'])==24*len(selected)
    close('group_sse',[r['sse']],[sse],2e-10);close('group_rmse',[r['rmse']],[rmse]);close('group_bias',[r['bias']],[math.fsum(v['bias'] for v in selected)/len(selected)]);assert int(r['good_days'])==sum(v['rmse']<=.1 for v in selected) and int(r['severe_days'])==sum(v['rmse']>=.2 for v in selected);close('day_fraction',[r['day_level_sse_fraction']],[math.fsum(v['sse_day'] for v in selected)/sse])
    bs=[v for k,v in daily.items() if k[0]=='BASE' and k[1]==r['seed'] and included(r['group'],k,v)];bsse=math.fsum(v['sse'] for v in bs);brmse=math.sqrt(bsse/(24*len(bs)));close('base_sse',[r['base_sse']],[bsse],2e-10);close('base_rmse',[r['base_rmse']],[brmse]);close('delta_sse',[r['delta_sse']],[sse-bsse],2e-10);close('rmse_percent',[r['rmse_percent']],[100*(rmse/brmse-1)],2e-10)
cases=read(D/'cases.csv');casekeys={(k[0],k[1],k[2],k[3]) for k in daily if (k[2],k[3]) in {('F47',160),('F47',161),('F13',98),('F13',112)}};assert len(cases)==32 and {(r['arm'],r['seed'],r['farm'],int(r['day'])) for r in cases}==casekeys
scorelookup={(r['arm'],r['seed'],r['farm'],int(r['day'])):r for r in scored}
for r in cases:assert r==scorelookup[(r['arm'],r['seed'],r['farm'],int(r['day']))]
changes=read(D/'day_changes.csv');assert len(changes)==1440
for r in changes:
    key=(R.ARM,r['seed'],r['farm'],int(r['day']));a=daily[key];b=daily[('BASE',*key[1:])]
    for n in ['truth','prediction','raw_et','sse','rmse']:close('changes_'+n,[r[n]],[a[n]],2e-11)
    close('changes_delta_sse',[r['delta_sse']],[a['sse']-b['sse']],2e-11);close('prediction_change',[r['prediction_change']],[a['prediction']-b['prediction']])
tags={};bias={str(s):daily[(R.ARM,str(s),'F47',161)]['raw_et']-daily[(R.ARM,str(s),'F47',161)]['truth'] for s in B.SEEDS}
tags['SELECTED_CASE_BIAS_REDUCED_ALL_SEEDS']=all(abs(bias[str(s)])<abs(daily[('BASE',str(s),'F47',161)]['raw_et']-daily[('BASE',str(s),'F47',161)]['truth']) for s in B.SEEDS);tags['RESIDUAL_BIAS_GT_POINT2_ALL_SEEDS']=all(v>.2 for v in bias.values());assert all(comp['tags'][n]==v for n,v in tags.items())
for s,v in bias.items():close('raw_et_tag_bias',[comp['tags']['raw_et_bias'][s]],[v])
# Independent ordered block reconstruction + exact RNG arrays + different sum primitive.
rng=np.random.default_rng(32617);totbase=np.zeros(20000);totalt=np.zeros(20000);totn=np.zeros(20000);block_checks=[]
with np.load(D/'bootstrap_draws.npz',allow_pickle=False) as z:
    for farm in ['F13','F47']:
        block=collections.defaultdict(list)
        for k,v in daily.items():
            if k[0]=='BASE' and k[1]=='ensemble' and k[2]==farm:block[k[3]//5].append(k[3])
        ids=sorted(block);vals=np.array([[math.fsum(daily[('BASE','ensemble',farm,d)]['sse'] for d in block[i]),math.fsum(daily[(R.ARM,'ensemble',farm,d)]['sse'] for d in block[i]),24*len(block[i])] for i in ids]);ix=rng.integers(0,len(ids),size=(20000,len(ids)),dtype=np.int32)
        assert np.array_equal(z[farm+'_index'],ix) and np.array_equal(z[farm+'_block'],ids);close('bootstrap_block_values',z[farm+'_sse'],vals,2e-10)
        # Sum sampled block multiplicities instead of summing indexed arrays as score does.
        count=np.zeros((20000,len(ids)),np.int32)
        for j in range(len(ids)):count[:,j]=(ix==j).sum(axis=1)
        totbase+=count@vals[:,0];totalt+=count@vals[:,1];totn+=count@vals[:,2];block_checks.append(dict(farm=farm,observed_blocks=len(ids),days=sum(map(len,block.values())),rows=int(vals[:,2].sum())))
    delta=100*(np.sqrt(totalt/totn)/np.sqrt(totbase/totn)-1);close('all20000_bootstrap_percent',z['rmse_change_percent'],delta,2e-10)
    saved=np.array([float(r['rmse_change_percent']) for r in read(D/'bootstrap_results.csv')]);assert len(saved)==20000;close('bootstrap_csv',saved,delta,2e-10)
    pw=float(np.mean(delta>=0));ci=np.percentile(delta,[2.5,97.5]);assert int((delta>=0).sum())==comp['bootstrap']['worse_or_equal_count'];close('p_worse',[comp['bootstrap']['p_worse']],[pw]);close('CI95',comp['bootstrap']['CI95_percent'],ci,2e-10)
allgain=all(math.fsum(v['sse'] for k,v in daily.items() if k[0]==R.ARM and k[1]==str(s))<math.fsum(v['sse'] for k,v in daily.items() if k[0]=='BASE' and k[1]==str(s)) for s in B.SEEDS)
guard=False
for s in B.SEEDS:
    bsse=math.fsum(v['sse'] for k,v in daily.items() if k[0]=='BASE' and k[1]==str(s) and k[3]>=179);asse=math.fsum(v['sse'] for k,v in daily.items() if k[0]==R.ARM and k[1]==str(s) and k[3]>=179);guard|=100*(math.sqrt(asse/bsse)-1)>=2
assert comp['DIAG_EXPANSION_SCREEN']==(allgain and pw<.025) and comp['PASS2_ADOPTION_HOLD_FLAG']==guard
for n,subset in [('ordinary_delta_sse',lambda k,v:v['truth']<1),('ordinary_remaining_delta_sse',lambda k,v:v['truth']<1 and (k[2],k[3])!=('F47',161)),('selected161_delta_sse',lambda k,v:(k[2],k[3])==('F47',161))]:
    a=math.fsum(v['sse'] for k,v in daily.items() if k[0]==R.ARM and k[1]=='ensemble' and subset(k,v));b=math.fsum(v['sse'] for k,v in daily.items() if k[0]=='BASE' and k[1]=='ensemble' and subset(k,v));close(n,[comp[n]],[a-b],2e-10)
inter=[]
for s in ['7','101','2024','ensemble']:
    a=rows[rows.arm.eq(R.ARM)&rows.seed.eq(s)].set_index('row_id');b=rows[rows.arm.eq('BASE')&rows.seed.eq(s)].set_index('row_id').loc[a.index]
    inter.append(dict(seed=s,gate_flips=int((a.gate!=b.gate).sum()),candidate_changes=int(((a.candidate_day.fillna(-1)!=b.candidate_day.fillna(-1))|(a.candidate_ec.fillna(-1)!=b.candidate_ec.fillna(-1))).sum()),max_sg2_interaction=float(abs((a.sg2_raw-a.pre_sg2)-(b.sg2_raw-b.pre_sg2)).max()),base_preclip=int((abs(b.smooth-b.pre_sg2)>1e-12).sum()),alt_preclip=int((abs(a.smooth-a.pre_sg2)>1e-12).sum()),base_postclip=int((abs(b.sg2_raw-b.prediction)>1e-12).sum()),alt_postclip=int((abs(a.sg2_raw-a.prediction)>1e-12).sum())))
for r,e in zip(read(D/'interactions.csv'),inter):
    for n,v in e.items():
        if n=='seed':assert r[n]==v
        else:close('interaction_'+n,[r[n]],[v])
mean=rows[rows.arm.eq(R.ARM)&rows.seed.ne('ensemble')].groupby('row_id').prediction.mean();actual=rows[rows.arm.eq(R.ARM)&rows.seed.eq('ensemble')].set_index('row_id').prediction;diff=actual-mean
out=dict(status='PASS_TWO_H0_ALL_CACHES_PIPELINE_SCORES_AND_BOOTSTRAP',fit=0,base_caches=30,alt_caches=30,rows=69120,day_scores=2880,group_scores=96,cases=32,max_errors=maxerr,bootstrap=dict(p_worse=pw,CI95_percent=ci.tolist(),repeats=20000,blocks=block_checks),tags=tags,raw_et_bias=bias,DIAG_EXPANSION_SCREEN=bool(allgain and pw<.025),PASS2_ADOPTION_HOLD_FLAG=bool(guard),interactions=inter,ensemble_vs_mean_post=dict(changed_rows=int((abs(diff)>1e-12).sum()),maxdiff=float(abs(diff).max()),mean_absdiff=float(abs(diff).mean())),cache_checks=cachechecks,scope='No new model fit/tree routing. Original immutable feature/season/SG2 functions reused; sums, RNG and scores independently reconstructed.')
with (H/'critic_verify_results_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(out['status'],out['bootstrap'],out['tags'],flush=True)
