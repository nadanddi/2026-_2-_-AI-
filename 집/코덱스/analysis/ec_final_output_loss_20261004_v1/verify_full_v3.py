"""Independent whole verifier for FINAL/RAW residual MLP. No runner import/refit.
Requires parent's pre-fit pinned verification_registration_v1.json and complete132.
NumPy checkpoint forward, independent moments, scalar/matrix causal formula,
fsum vs NumPy metrics, manual bootstrap quantiles. Only public labels/train_X.
"""
from pathlib import Path
import sys, json, ast, math, hashlib, platform, argparse
sys.dont_write_bytecode=True
assert not sys.flags.optimize
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(H));import verification_math_v4 as M
np,pd=M.np,M.pd
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
sys.path.insert(0,str(H.parent/'statistical_experiments_20261003_v1'));import support as S
OUT=ROOT/'집/코덱스/local'/H.name
INNER=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
IH=H.parent/'ec_matched_inner_calibration_20261003_v1'
NPATH=H.parent/'ec_nested_high_specialist_20261004_v1/run_v2.py'
OLD=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
BASE=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1'
GUARD=H.parent/'ec_tabdpt_20261004_v1/verify_full_v2.py'
MODES=('FINAL_LOSS','RAW_LOSS');SEEDS=(7,101,2024)
FOLDS=[(v,k) for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for k in range(n)]
COLS=['row_id','farm','day','hour','y','baseline','candidate','base_raw','delta','validator','fold','seed','clip_lo','clip_hi','mode']
PINS=dict(support='82074e7a04ea681bc94e94e1c90354a743609e0f008b1a378a5a9a23575debed',core='057ff4d8da6f3af29105b049251498f79f7fcd6b88d980a8222e5629eb5ce3b2',env='82ca7b4bda2e9f50069b53b92d8418a7c0d4dd5ce7127fade16dd6ea78a8dcd2',env_extra='8b83d9e2651b3ac2787db59042267951f42b7f5e098a91d90785dc81dd6cbffb',season='7d58feeb6a0e7653796a6775f762b659cab3e64078b299d36b52ae69faeff162',nested_source='0ceb89e38692f76466000a4b3a83e056c99a04df10c519ddba59e70a55c2bbaf',matched_source='b1041713e093ca9b8f17437e05b194f284a7e44e8a681a06ce033bc6fc8a7e40',original_guard='e5398b7c29eae32a4efc75381f7478cb1937e28bb4e61db1c1989016d732d9d9')

def sha(p):return M.sha(p)
def readj(p):
    def pairs(v):
        d={}
        for k,x in v:assert k not in d;d[k]=x
        return d
    return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=pairs,
       parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def ah(x):
    x=np.asarray(x);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def ids(x):return hashlib.sha256('\n'.join(map(str,x)).encode()).hexdigest()
def extract(path,names,ns):
    a=[n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in a}==set(names) and len(a)==len(names)
    exec(compile(ast.Module(body=a,type_ignores=[]),str(path),'exec'),ns);return ns
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def frames_equal(a,b):
    assert list(a.columns)==list(b.columns)==COLS and len(a)==len(b)
    for c in COLS:
        if c in ['row_id','farm','day','hour','validator','fold','seed','mode']:assert np.array_equal(a[c],b[c])
        else:M.near(a[c],b[c])
def first_check(f,sig,q):
    assert set(f)=={'status','signature','atol','epochs','errors','initial_weights_sha256','all_gradients_finite','all_parameters_finite','zero_query_baseline_maxdiff','zero_inner_baseline_maxdiff','scalar_maxdiff','fresh_loss_trace_maxdiff','prefix','loss_initial','loss_final','limitations'}
    assert f['status']=='PASS' and f['signature']==sig and f['atol']==1e-12 and f['epochs']==400
    assert f['all_gradients_finite'] is True and f['all_parameters_finite'] is True
    assert f['initial_weights_sha256']==sig['initial_weights_sha256'][str(sig['seed'])]
    assert set(f['errors'])=={'repeat','fresh','reversed','single','other_query'}
    for x in list(f['errors'].values())+[f[k] for k in ['zero_query_baseline_maxdiff','zero_inner_baseline_maxdiff','scalar_maxdiff','fresh_loss_trace_maxdiff']]:
        assert isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=1e-12
    expected=[(farm,int(q.loc[q.farm==farm,'day'].min()),h) for farm in ('F13','F47') for h in (0,6,12)]
    assert [(r['farm'],r['day'],r['hour']) for r in f['prefix']]==expected
    for r in f['prefix']:
        assert r['rows']==r['hour']+1
        for c in ['raw_delta_maxdiff','final_maxdiff']:assert math.isfinite(r[c]) and 0<=r[c]<=1e-12

def verify():
    receipt=readj(H/'verification_registration_v1.json')
    assert receipt['status']=='PINNED_BEFORE_ACTUAL_FIT'
    mandatory=[H/'run_v3.py',H/'preparation_v3.json',H/'preregistration_v1.md',Path(__file__),Path(M.__file__)]
    assert set(str(p.relative_to(ROOT)) for p in mandatory)<=set(receipt['files'])
    assert receipt['runner_file']=='run_v3.py' and receipt['preparation_file']=='preparation_v3.json'
    destinations=['full_scores_v3.csv','full_segments_v3.csv','training_and_outer_loss_v3.csv','matched_loss_comparison_v3.csv','full_verification_v3.json']
    assert all(not (H/p).exists() for p in destinations)
    for rel,h in receipt['files'].items():assert sha(ROOT/rel)==h,rel
    prep=readj(H/receipt['preparation_file']);fit=readj(H/'fit_audit_v1.json')
    assert fit['status']=='PASS' and fit['preparation']==prep and fit['score_count']==0
    assert len(fit['cells'])==132 and fit['cells_per_mode']==66 and fit['rows']==166320
    assert prep['status']=='PASS_PREPARATION_MODEL_FIT0_PREDICT0_SCORE0'
    assert all(prep[k]==0 for k in ['model_fit_count','model_predict_count','real_score_count'])
    assert prep['required_cpu']==22 and prep['required_pfn']==88
    assert prep['source_sha256']==sha(H/receipt['runner_file'])
    assert prep['config']==fit['config'] and prep['config']['alpha']==M.ALPHA
    assert hashlib.sha256(json.dumps(prep['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==receipt['config_sha256']
    assert prep['config']['epochs']==400 and prep['config']['hidden']==[64,32] and prep['config']['lr']==.001 and prep['config']['weight_decay']==.01
    assert prep['config']['strict_15'] and prep['config']['dtype']=='float64'
    lab,core,wv,folds,outer=S.loadec();assert [(v,k) for v,k,_,_ in folds]==FOLDS
    import sklearn,lightgbm,torch
    rt=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,torch=torch.__version__)
    assert rt==prep['runtime']['versions']
    for name,mod in [('numpy',np),('pandas',pd),('sklearn',sklearn),('lightgbm',lightgbm),('torch',torch)]:
        assert str(Path(mod.__file__).resolve())==prep['runtime']['module_paths'][name]
        assert sha(Path(mod.__file__))==prep['runtime']['module_init_sha256'][name]
    deps=dict(run=sha(H/receipt['runner_file']),support=sha(Path(S.__file__)),core=sha(Path(core.__file__)),env=sha(Path(env.__file__)),env_extra=sha(Path(env_extra.__file__)),season=sha(Path(sys.modules[S.mapping.__module__].__file__)),nested_source=sha(NPATH),matched_source=sha(IH/'run.py'),original_guard=sha(GUARD))
    assert {k:v for k,v in deps.items() if k!='run'}==PINS and deps==prep['dependencies']
    assert prep['inputs']==dict(train_X=sha(Path(env.DATA)/'train_X.csv'),public_oof=sha(OLD/'v2_integration_oof.csv'))
    assert prep['inputs']['train_X']=='21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300'
    cols=[c for c in core.BASE if c!='day']+['season'];assert len(cols)==14 and prep['features']==cols
    assert not set(cols)&{'in_rad','act_side','act_valve','act_cool','act_pump'}
    n=extract(NPATH,{'arrayfile','full_inner'},dict(np=np,S=S,INNER=INNER))['full_inner']
    g=extract(GUARD,{'guard_original_r3'},dict(np=np,json=json,hashlib=hashlib,compare=M.near,ids_sha=ids,array_sha=ah))['guard_original_r3']
    idx=lab.set_index('row_id');labels=lab.set_index('row_id').sub_ec
    assert len(lab)==8640 and idx.index.is_unique and len(prep['manifest'])==22
    refs={};r3audits=[];metas=[];conc={m:[] for m in MODES};loss_records=[];checkpoint_gap=0.;prefix_gap=0.
    for j,(v,k,tm,vm) in enumerate(folds):
        sig=prep['manifest'][j];assert (sig['validator'],sig['fold'])==(v,k)
        assert sig['dependencies']==deps and sig['runtime']==prep['runtime'] and sig['config']==prep['config'] and sig['inputs']==prep['inputs'] and sig['features']==cols
        for rel,h in sig['cache_hashes'].items():assert sha(ROOT/rel)==h
        a,b,z,bag=n(v,k,lab[tm].reset_index(drop=True),idx)
        a,b=S.seasonal(a,b,wv);tr,q=S.seasonal(lab[tm],lab[vm],wv)
        a,b,tr,q=[x.reset_index(drop=True) for x in (a,b,tr,q)]
        for name,f in [('inner_a',a),('inner_b',b),('outer_train',tr),('outer_query',q)]:
            assert sig[name+'_ids']==ids(f.row_id) and sig[name+'_rows']==len(f) and sig[name+'_days']==len(f[['farm','day']].drop_duplicates())
            assert sig[name+'_features']==ah(f[cols].to_numpy(float)) and sig[name+'_targets']==ah(f.sub_ec.to_numpy(float))
            assert all(len(h)==24 and set(h.hour)==set(range(24)) for _,h in f.groupby(['farm','day']))
        assert set(a.row_id)|set(b.row_id)<=set(tr.row_id) and not (set(a.row_id)|set(b.row_id))&set(q.row_id)
        ip=[float(a.sub_ec.min()),float(a.sub_ec.max())];op=[float(tr.sub_ec.min()),float(tr.sub_ec.max())]
        M.near(ip,sig['inner_bounds']);M.near(op,sig['outer_bounds'])
        old=arrays(BASE/f'{v}_{k}_baseline.npz');assert np.array_equal(old['row_id'],q.row_id)
        M.near([old['lo'],old['hi']],op)
        pfn=[]
        for c in (1,2,3,4):
            pp=arrays(OLD/f'{v}_{k}_pfn_{c}.npz');assert np.array_equal(pp['row_id'],q.row_id)
            assert np.array_equal(pp['context_row_id'],old[f'context_{c}']) and set(pp['context_row_id'])<=set(tr.row_id)
            M.near(pp['sub_ec'],q.sub_ec);pfn.append(pp['raw_pfn'])
        M.near(old['old_pfn_raw'],np.mean(pfn,axis=0))
        for seed in SEEDS:
            rpath=OLD/f'{v}_{k}_r3_{seed}.npz';rz=arrays(rpath);rm=readj(rpath.with_suffix('.json'))
            prov=rm['provenance']
            assert prov['feature_order']=='remove day; append season'
            assert prov['shared']['input_sha256']['train_X.csv']==prep['inputs']['train_X'] and prov['shared']['core_sha256']==deps['core']
            assert all(prov['environment'][key]==rt[key] for key in ['python','numpy','pandas','sklearn','lightgbm'])
            r3audits.append(g(rz,rm,sha(rpath),tr,q,seed,v,k,labels))
            M.near(old[f'r3_{seed}'],rz['raw_r3'])
            base=.8*rz['raw_r3']+.2*old['old_pfn_raw'];innerraw=.8*z[f'r3_{seed}']+.2*bag
            baseline=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(q.row_id).to_numpy()
            assert ah(innerraw)==sig['inner_base_raw'][str(seed)] and ah(base)==sig['outer_base_raw'][str(seed)] and ah(baseline)==sig['outer_baseline'][str(seed)]
            M.near(baseline,old[f'baseline_{seed}']);M.near(baseline,M.scalar_final(q,base,*op))
            for mode in MODES:
                name=f'{mode}_{v}_{k}_{seed}';csv=OUT/(name+'_pred.csv');cp=OUT/(name+'_model.npz');meta=readj(OUT/(name+'.json'))
                expected=dict(**sig,seed=seed,mode=mode,family=22 if mode=='FINAL_LOSS' else 23,preparation_sha256=sha(H/receipt['preparation_file']),preregistration_sha256=sha(H/'preregistration_v1.md'))
                assert meta['status']=='PASS' and meta['signature']==expected and meta['csv_sha256']==sha(csv) and meta['checkpoint_sha256']==sha(cp)
                assert set(meta)=={'status','signature','csv_sha256','checkpoint_sha256','train_seconds','train'}|({'first_audit_sha256'} if (v,k,seed)==('DIAG10',0,7) else set())
                assert math.isfinite(meta['train_seconds']) and meta['train_seconds']>=0
                d=pd.read_csv(csv,float_precision='round_trip');assert list(d.columns)==COLS and len(d)==len(q) and np.array_equal(d.row_id,q.row_id)
                for c in ['farm','day','hour']:assert np.array_equal(d[c],q[c])
                for c,x in [('mode',mode),('validator',v),('fold',k),('seed',seed)]:assert (d[c]==x).all()
                for c in ['day','hour','fold','seed']:assert pd.api.types.is_integer_dtype(d[c].dtype)
                M.near(d.y,q.sub_ec);M.near(d.baseline,baseline);M.near(d.base_raw,base)
                M.near(d.clip_lo,np.full(len(d),op[0]));M.near(d.clip_hi,np.full(len(d),op[1]))
                weights=arrays(cp);assert str(weights['format'])=='RESIDUAL_MLP_FLOAT64_V1'
                assert set(weights)==set(sig['preprocessing'])|{'feature_names','format','loss_trace'}|{f'model__{l}.{p}' for l in (0,2,4) for p in ('weight','bias')}
                for key in sig['preprocessing']:assert ah(weights[key])==sig['preprocessing'][key]
                def stored_transform(f):
                    xx=f[cols].to_numpy(float).copy();missing=np.isnan(xx)
                    xx[missing]=np.broadcast_to(weights['imputer_statistics'],xx.shape)[missing]
                    return (xx-weights['scaler_mean'])/weights['scaler_scale']
                assert ah(stored_transform(b))==sig['inner_b_transformed_features']
                assert ah(stored_transform(q))==sig['outer_query_transformed_features']
                delta=M.replay_checkpoint(weights,b[cols].to_numpy(float),q[cols].to_numpy(float),cols)
                checkpoint_gap=max(checkpoint_gap,M.near(d.delta,delta))
                final=M.scalar_final(q,base+delta,*op);prefix_gap=max(prefix_gap,M.near(d.candidate,final))
                M.near(final,M.matrix_final(q,base+delta,*op))
                dt=M.replay_checkpoint(weights,b[cols].to_numpy(float),b[cols].to_numpy(float),cols)
                y=b.sub_ec.to_numpy(float)
                if mode=='FINAL_LOSS':initial=M.scalar_final(b,innerraw,*ip);last=M.scalar_final(b,innerraw+dt,*ip)
                else:initial=innerraw;last=innerraw+dt
                trace=weights['loss_trace'];assert trace.shape==(401,) and np.isfinite(trace).all()
                loss0=math.fsum((float(p)-float(t))**2 for p,t in zip(initial,y))/len(y)
                loss1=math.fsum((float(p)-float(t))**2 for p,t in zip(last,y))/len(y)
                M.near([trace[0],trace[-1]],[loss0,loss1]);mt=meta['train']
                assert set(mt)=={'initial_weights_sha256','all_gradients_finite','all_parameters_finite','epochs','loss_initial','loss_final','loss_trace_sha256','training_rows','training_days','inner_bounds','outer_bounds','config'}
                assert mt['epochs']==400 and mt['training_rows']==len(b) and mt['training_days']==len(b[['farm','day']].drop_duplicates()) and mt['config']==prep['config']
                assert mt['all_gradients_finite'] is True and mt['all_parameters_finite'] is True
                assert mt['initial_weights_sha256']==sig['initial_weights_sha256'][str(seed)]
                M.near(mt['inner_bounds'],ip);M.near(mt['outer_bounds'],op)
                assert mt['loss_trace_sha256']==ah(trace);M.near([mt['loss_initial'],mt['loss_final']],[loss0,loss1])
                loss_records.append(dict(mode=mode,validator=v,fold=k,seed=seed,train_rows=len(b),train_loss_initial=loss0,train_loss_final=loss1))
                if (v,k,seed)==('DIAG10',0,7):
                    fp=H/f'first_{mode}_v1.json';assert meta['first_audit_sha256']==sha(fp);first_check(readj(fp),expected,q)
                    M.near([readj(fp)['loss_initial'],readj(fp)['loss_final']],[loss0,loss1])
                else:assert 'first_audit_sha256' not in meta
                refs[mode,v,k,seed]=meta;conc[mode].append(d)
    assert r3audits==prep['original_r3_guard']
    expectedmeta=[refs[m,v,k,s] for m in MODES for v,k in FOLDS for s in SEEDS]
    assert fit['cells']==expectedmeta
    tables={}
    for mode in MODES:
        tables[mode]=pd.concat(conc[mode],ignore_index=True);d=tables[mode]
        assert len(d)==83160 and not d.duplicated(['validator','fold','seed','row_id']).any()
        frames_equal(d,pd.read_csv(OUT/f'oof_{mode}.csv',float_precision='round_trip'))
        assert fit['mode_aggregate_sha256'][mode]==sha(OUT/f'oof_{mode}.csv')
    frames_equal(pd.concat([tables[m] for m in MODES],ignore_index=True),pd.read_csv(OUT/'oof.csv',float_precision='round_trip'))
    assert fit['aggregate_sha256']==sha(OUT/'oof.csv')
    # No outer scores above this complete output/provenance/checkpoint gate.
    for record in loss_records:
        d=tables[record['mode']];cell=d[(d.validator==record['validator'])&(d.fold==record['fold'])&(d.seed==record['seed'])]
        record['outer_rmse']=M.rmse(cell.y,cell.candidate)
    scores=[];boots={};passed={};segments=[];comparison=[]
    for mode,d in tables.items():
        boots[mode]={};ok=True
        for (v,seed),g in d.groupby(['validator','seed'],sort=False):
            rb,rc=M.rmse(g.y,g.baseline),M.rmse(g.y,g.candidate);ok &= rc<rb
            assert (rc<rb)==(float(np.mean((g.y-g.candidate)**2))<float(np.mean((g.y-g.baseline)**2)))
            foldrmse=[M.rmse(t.y,t.candidate) for _,t in g.groupby('fold')]
            scores.append(dict(mode=mode,validator=v,seed=int(seed),n=len(g),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1),fold_rmse_sd=float(np.std(foldrmse,ddof=1)) if len(foldrmse)>1 else None))
            if v=='DIAG10':
                boot=M.bootstrap(g,int(seed));boots[mode][str(seed)]=boot;ok &= boot['pass_gate']
                daily=g.groupby(['farm','day']).y.mean();high=set(daily[daily>=1].index);assert len(high)==31
                subsets=dict(high=g[np.array([(f,int(day)) in high for f,day in zip(g.farm,g.day)])],ordinary=g[np.array([(f,int(day)) not in high for f,day in zip(g.farm,g.day)])],late=g[g.day>=179],F13=g[g.farm=='F13'],F47=g[g.farm=='F47'],hour0=g[g.hour==0])
                for label,t in subsets.items():segments.append(dict(mode=mode,seed=int(seed),segment=label,n=len(t),days=len(t[['farm','day']].drop_duplicates()),baseline=M.rmse(t.y,t.baseline),candidate=M.rmse(t.y,t.candidate),baseline_bias=math.fsum(t.baseline-t.y)/len(t),candidate_bias=math.fsum(t.candidate-t.y)/len(t)))
        passed[mode]=bool(ok)
    final=tables['FINAL_LOSS'];raw=tables['RAW_LOSS'];keys=['validator','fold','seed','row_id']
    assert np.array_equal(final[keys],raw[keys]);M.near(final.y,raw.y);M.near(final.baseline,raw.baseline)
    for (v,s),g in final.groupby(['validator','seed'],sort=False):
        r=raw[(raw.validator==v)&(raw.seed==s)];a,b=M.rmse(g.y,g.candidate),M.rmse(r.y,r.candidate)
        comparison.append(dict(validator=v,seed=int(s),final_rmse=a,raw_loss_rmse=b,change_pct=100*(a/b-1)))
    pd.DataFrame(scores).to_csv(H/'full_scores_v3.csv',index=False,mode='x')
    pd.DataFrame(segments).to_csv(H/'full_segments_v3.csv',index=False,mode='x')
    pd.DataFrame(loss_records).to_csv(H/'training_and_outer_loss_v3.csv',index=False,mode='x')
    pd.DataFrame(comparison).to_csv(H/'matched_loss_comparison_v3.csv',index=False,mode='x')
    report=dict(status='PASS_WHOLE_NUMPY_CHECKPOINT_AND_PUBLIC_ARITHMETIC',candidate_pass=passed,cells=132,cells_per_mode=66,rows=166320,score_cells_per_mode=15,alpha=M.ALPHA,bootstrap=boots,checks=M.CHECKS,checkpoint_maxdiff=checkpoint_gap,scalar_prefix_maxdiff=prefix_gap,verification_fit=0,checkpoint_replays=264,raw_EC_reads=0,test_reads=0,EL1_rescore=0,receipt_sha256=sha(H/'verification_registration_v1.json'),verifier_sha256=sha(Path(__file__)),fit_audit_sha256=sha(H/'fit_audit_v1.json'),outputs_sha256={p:sha(H/p) for p in destinations if p!='full_verification_v3.json'},limitations=['Repeated public validation, not new independent holdout','Inner residual training only heldout subset; training loss can differ from outer generalization','Native first-audit receipt verified, no independent full optimizer training replay'])
    save(H/'full_verification_v3.json',report)
    print('PASS_WHOLE',passed,M.CHECKS,checkpoint_gap,prefix_gap)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    assert args.verify,'Run only after parent verifies complete output132'
    verify()
