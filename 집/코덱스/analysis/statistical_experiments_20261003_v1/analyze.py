from support import *
import time

def prep_z(a,b):
    prep=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler());x=prep.fit_transform(a);v=prep.transform(b)
    return np.column_stack([np.ones(len(x)),x]),np.column_stack([np.ones(len(v)),v])
def bounded_weights(proposed,old,g):
    lo=np.maximum(0,old-.1*g[:,None]);hi=np.minimum(1,old+.1*g[:,None]);left=np.full(len(g),-2.);right=np.full(len(g),2.)
    for _ in range(60):
        mid=(left+right)/2;total=np.clip(proposed-mid[:,None],lo,hi).sum(1);left=np.where(total>1,mid,left);right=np.where(total>1,right,mid)
    answer=np.clip(proposed-((left+right)/2)[:,None],lo,hi);assert np.max(np.abs(answer.sum(1)-1))<1e-12;return answer
def fitmeta(z,p,outerp,qz,qg):
    g=z['gate'];old=np.column_stack([.4+.1*g,.6-.4*g,.3*g]);base=(old*p).sum(1);qold=np.column_stack([.4+.1*qg,.6-.4*qg,.3*qg]);ref=(qold*outerp).sum(1);x,v=prep_z(z['z'],qz)
    bias=Ridge(alpha=100.,fit_intercept=False).fit(x,z['y']-base,sample_weight=z['w']);bc=qg*np.clip(v@bias.coef_,-.35,.35)
    d=np.column_stack([x*(g*(p[:,0]-p[:,2]))[:,None],x*(g*(p[:,1]-p[:,2]))[:,None]])
    reg=Ridge(alpha=1000.,fit_intercept=False).fit(d,z['y']-base,sample_weight=z['w']);size=x.shape[1];a=qg*(v@reg.coef_[:size]);b=qg*(v@reg.coef_[size:]);proposed=qold+np.column_stack([a,b,-a-b]);qw=bounded_weights(proposed,qold,qg)
    return ref+bc,(qw*outerp).sum(1),dict(bias_coef=bias.coef_.tolist(),gate_coef=reg.coef_.tolist(),bias_delta=bc,weights=qw,old_weights=qold,cal_base=base,cal_design=d,cal_z=x,cal_target=z['y']-base,cal_w=z['w'],outer_z=v,outer_p=outerp,outer_g=qg)
def pfn(prefix,seeds,row_id):
    ps=[]
    for seed in seeds:
        path=OUT/f'{prefix}_pfn_{seed}.npz';z=dict(np.load(path));assert np.array_equal(z['row_id'],row_id);assert str(z['input_hash'])==sha(OUT/f'{prefix}_input.npz');ps.append(z['prediction'])
    return np.mean(ps,axis=0)
def evaluate(oof):
    scores=[]
    for (target,arm,name,seed,context),d in oof.groupby(['target','arm','validator','seed','context'],sort=True):
        y=d.y.to_numpy();a=d.baseline.to_numpy();b=d.candidate.to_numpy();r0=float(np.sqrt(np.mean((a-y)**2)));r1=float(np.sqrt(np.mean((b-y)**2)));ff=[float(np.sqrt(np.mean((g.candidate-g.y)**2))) for _,g in d.groupby('fold')]
        rec=dict(target=target,arm=arm,validator=name,seed=int(seed),context=context,n=len(d),baseline_rmse=r0,candidate_rmse=r1,delta_pct=100*(r1/r0-1),candidate_fold_mean=float(np.mean(ff)),candidate_fold_sd=float(np.std(ff,ddof=1)) if len(ff)>1 else None)
        if name=='DIAG10':
            d=d.copy();d['block5']=d.day//5;d['mse0']=(d.baseline-d.y)**2;d['dmse']=(d.candidate-d.y)**2-d.mse0;g=d.groupby(['farm','block5'],sort=True).agg(n=('y','size'),delta=('dmse','sum'));rng=np.random.default_rng(20261003);ix=[]
            farms=g.index.get_level_values(0)
            for farm in ['F13','F47']:
                ids=np.flatnonzero(farms==farm);ix.append(ids[rng.integers(0,len(ids),(20000,len(ids)))])
            ix=np.concatenate(ix,axis=1);samples=g.delta.to_numpy()[ix].sum(1)/g.n.to_numpy()[ix].sum(1);rec.update(p_worse=float(np.mean(samples>=0)),ci_mse=np.quantile(samples,[ALPHA,1-ALPHA]).tolist(),alpha=ALPHA,n_blocks=len(g))
        scores.append(rec)
    decisions={}
    for arm in ['TBIAS','TGATE','ESTATE']:
        rows=[r for r in scores if r['arm']==arm];assert len(rows)==(15 if arm=='ESTATE' else 12)
        passed=all(r['delta_pct']<0 for r in rows) and all(r['p_worse']<ALPHA and r['ci_mse'][1]<0 for r in rows if r['validator']=='DIAG10')
        decisions[arm]='PASS_REQUIRES_GUARD_EL1' if passed and arm!='ESTATE' else 'PUBLIC_PASS_NO_NEW_LOCK' if passed else 'REJECT'
    return scores,decisions
def main():
    start=time.time()
    while not (OUT/'gpu_done.json').exists() or not (OUT/'plan.json').exists():time.sleep(3)
    lab,_,_,_,folds,outer=loadtemp();records=[];diagnostics=[]
    for name,k,fd in folds:
        tm,vm=common.split_mask(lab,fd);va=lab[vm].copy();prefix=f'T_{name}_{k}';z=dict(np.load(OUT/f'{prefix}_cpu.npz'))
        for seed in TSEEDS:
            for context,seeds in [('1-8',list(range(1,9))),('17-24',list(range(17,25)))]:
                p=np.column_stack([z[f'base_{seed}'],z[f'codex_{seed}'],pfn(prefix,seeds,z['row_id'])]);d=outer[(outer.validator==name)&(outer.base_seed==seed)&(outer.context==context)];values=[]
                for member in ['BASE','CODEX','PFN','W30G']:values.append(d[d.member==member].set_index('row_id').prediction.reindex(va.row_id).to_numpy())
                qp=np.column_stack(values[:3]);ref=values[3];assert np.isfinite(qp).all();assert np.max(np.abs(ref-(weights(va)*qp).sum(1)))<1e-12
                bias,gatep,audit=fitmeta(z,p,qp,conditions(va),gate(va));fp=OUT/f'{prefix}_{seed}_{context}_meta.npz'
                if not fp.exists():np.savez(fp,**{q:np.asarray(v) for q,v in audit.items()},cal_p=p,cal_y=z['y'],outer_row_id=va.row_id.to_numpy(dtype=str))
                for arm,pred in [('TBIAS',bias),('TGATE',gatep)]:
                    f=va[['row_id','farm','day','hour']].copy();f['y']=va.sub_temp.to_numpy();f['target']='TEMP';f['arm']=arm;f['validator']=name;f['fold']=k;f['seed']=seed;f['context']=context;f['baseline']=ref;f['candidate']=pred;records.append(f)
        print(f'{prefix} meta evaluation done',flush=True)
    lab,core,wv,folds,outer=loadec()
    for name,k,tm,vm in folds:
        tr,va=lab[tm].copy(),lab[vm].copy();prefix=f'E_{name}_{k}';z=dict(np.load(OUT/f'{prefix}_cpu.npz'));b=pd.DataFrame(dict(row_id=z['row_id'],farm=z['farm'],day=z['day'],hour=z['hour'],sub_ec=z['y']));a=pd.DataFrame(dict(row_id=z['observed_id'],sub_ec=z['observed_y']));a['farm']=a.row_id.str[:3];a['day']=a.row_id.str[4:7].astype(int)
        bag=pfn(prefix,[1,2,3,4],z['row_id'])
        for seed in ESEEDS:
            innerp=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi']);x,gap,source=state_features(a,b,innerp);reg=Ridge(alpha=100.,fit_intercept=False).fit(x,z['y']-innerp)
            d=outer[(outer.validator==name)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').reindex(va.row_id);assert np.isfinite(d.season_v2).all();ref=d.season_v2.to_numpy();v,qgap,qsrc=state_features(tr,va,ref);vs,sgap,ssrc=state_features(tr,va,ref,delay=5)
            delta=np.clip(v@reg.coef_,-.15,.15);pred=np.clip(ref+delta,tr.sub_ec.min(),tr.sub_ec.max());stress=np.clip(ref+np.clip(vs@reg.coef_,-.15,.15),tr.sub_ec.min(),tr.sub_ec.max())
            # Future and other-farm observations cannot change the query's state.
            for farm in ['F13','F47']:
                if not (va.farm==farm).any():continue
                cut=va[va.farm==farm].sort_values(['day','hour']).iloc[0];q=va[(va.farm==farm)&(va.day==cut.day)];pr=ref[(va.farm==farm)&(va.day==cut.day)];changed=tr.copy();bad=(changed.farm!=farm)|(changed.day>=cut.day);changed.loc[bad,'sub_ec']=changed.loc[bad,'sub_ec']*7+99;v0,_,_=state_features(tr,q,pr);v1,_,_=state_features(changed,q,pr);assert np.array_equal(v0,v1)
            mp=OUT/f'{prefix}_{seed}_meta.npz'
            if not mp.exists():np.savez(mp,cal_design=x,cal_target=z['y']-innerp,coef=reg.coef_,outer_design=v,outer_source=qsrc,outer_gap=qgap,inner_source=source,inner_gap=gap,outer_row_id=va.row_id.to_numpy(dtype=str),clip_lo=tr.sub_ec.min(),clip_hi=tr.sub_ec.max(),stress=stress)
            f=va[['row_id','farm','day','hour']].copy();f['y']=va.sub_ec.to_numpy();f['target']='EC';f['arm']='ESTATE';f['validator']=name;f['fold']=k;f['seed']=seed;f['context']='1-4';f['baseline']=ref;f['candidate']=pred;f['stress_delay5']=stress;f['past_gap']=qgap;records.append(f)
            good=np.isfinite(gap);r=z['y']-innerp
            diagnostics.append(dict(validator=name,fold=k,seed=seed,inner_residual_lastgap_correlation=float(np.corrcoef(x[good,0],r[good])[0,1]) if good.sum()>2 and np.std(x[good,0])>0 else None,coef=reg.coef_.tolist(),inner_residual_rmse=float(np.sqrt(np.mean(r*r))),inner_after_rmse=float(np.sqrt(np.mean((r-x@reg.coef_)**2))),outer_no_past=int(np.isnan(qgap).sum()),causal_invariance=True))
        print(f'{prefix} state evaluation done',flush=True)
    oof=pd.concat(records,ignore_index=True);oof.to_csv(OUT/'oof.csv',index=False);scores,decisions=evaluate(oof)
    pd.DataFrame(scores).to_csv(HERE/'scores.csv',index=False);pd.DataFrame(diagnostics).to_csv(HERE/'ec_residual_diagnostics.csv',index=False)
    segments=[]
    for (target,arm,name,seed,context),d in oof.groupby(['target','arm','validator','seed','context']):
        if name!='DIAG10':continue
        for tag,mask in [('early',d.day<179),('late',d.day>=179),('F13',d.farm=='F13'),('F47',d.farm=='F47'),('F47_late',(d.farm=='F47')&(d.day>=179))]:
            q=d[mask];segments.append(dict(target=target,arm=arm,seed=int(seed),context=context,segment=tag,n=len(q),baseline_rmse=float(np.sqrt(np.mean((q.baseline-q.y)**2))),candidate_rmse=float(np.sqrt(np.mean((q.candidate-q.y)**2)))))
    pd.DataFrame(segments).to_csv(HERE/'segments.csv',index=False)
    plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'));sx=pd.read_csv(Path(env.DATA)/'test_X.csv',usecols=['row_id']);sx=sx[sx.row_id.str[:3].isin(['F13','F47'])];sx['farm']=sx.row_id.str[:3];sx['day']=sx.row_id.str[4:7].astype(int)
    tl,_,_,_,_,_=loadtemp();el,_,_,_,_=loadec();gap=dict(inner_outer=plan['audits'],actual_evaluation_metadata_only=dict(TEMP=gapstats(tl,sx),EC=gapstats(el,sx)))
    savej(HERE/'gap_validation.json',gap);savej(HERE/'result.json',dict(decisions=decisions,scores=scores,diagnostics=diagnostics,elapsed=time.time()-start,family_k=K,alpha=ALPHA,source_hashes={p.name:sha(p) for p in HERE.glob('*.py')},oof_hash=sha(OUT/'oof.csv')))
    print(json.dumps(decisions),flush=True)
if __name__=='__main__':main()
