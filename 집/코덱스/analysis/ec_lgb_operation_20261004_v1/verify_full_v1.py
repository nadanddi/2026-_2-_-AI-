"""Independent full-output verifier for family21 LGB-only operation features.

Draft source pins are replaced in a new version before the first model fit.
No model fit/predict or TabDPT runtime calls are made here.
"""
from pathlib import Path
import sys, json, math, hashlib, ast, argparse, importlib.util
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
import pandas as pd
V_PATH = H.parent/'ec_tabdpt_20261004_v1/verify_full_v2.py'
V_SHA = 'e5398b7c29eae32a4efc75381f7478cb1937e28bb4e61db1c1989016d732d9d9'
EXPECTED_RUN_FILE = 'run_v1.py'
EXPECTED_RUN_SHA = 'ROOT_TO_PIN_BEFORE_FIT'
EXPECTED_PREP_FILE = 'preparation_v1.json'
EXPECTED_PREP_SHA = 'ROOT_TO_PIN_BEFORE_FIT'
DP1 = ROOT/'집/클로드/research/ec3_DP1_daily_operation_pattern_v1.py'
DP1_SHA = '40d650c639550c09ef96a6be3aae3ef0f969fa1cbd899c787e7724a7f563e86d'
OUT = ROOT/'집/코덱스/local'/H.name
OLD = ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
BASE = ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1'
SEEDS = (7,101,2024)
FAMILY = 21
ALPHA = .025/FAMILY
NEW = ['seal_run','vent_open_hours','first_open_hour','thermal_switches','shade_switches',
       'since_curtain_change','heat_run','co2_hours','vent_max']
CHECKS = 0

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def compare(a,b,tol=1e-12):
    global CHECKS
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    e=float(np.max(np.abs(a-b)))
    assert e<=tol,e
    CHECKS+=a.size
    return e

def scalar_candidate(q,et,new_lgb,mlp,pfn,lo,hi):
    # Independent flattened coefficients and scalar farm/day prefix accumulation.
    values={};result=np.empty(len(q))
    for i in sorted(range(len(q)),key=lambda i:(q.farm.iloc[i],int(q.day.iloc[i]),int(q.hour.iloc[i]))):
        raw=math.fsum([.48*float(et[i]),.24*float(new_lgb[i]),.08*float(mlp[i]),.2*float(pfn[i])])
        prior=values.setdefault((q.farm.iloc[i],int(q.day.iloc[i])),[])
        prior.append(raw)
        result[i]=min(hi,max(lo,.5*raw+.5*math.fsum(prior)/len(prior)))
    return result

def boot(d,seed):
    d=d.copy()
    d['loss_diff']=(d.y-d.candidate)**2-(d.y-d.baseline)**2
    daily=d.groupby(['farm','day'],sort=True).loss_diff.agg(['sum','count'])
    rng=np.random.default_rng(20261003+seed)
    total=np.zeros(20000);count=np.zeros(20000)
    for farm in ['F13','F47']:
        g=daily.loc[farm].sort_index()
        blocks=[g.iloc[i:i+5] for i in range(0,len(g),5)]
        a=np.array([x['sum'].sum() for x in blocks]);n=np.array([x['count'].sum() for x in blocks])
        ix=rng.integers(len(a),size=(20000,len(a)))
        total+=a[ix].sum(axis=1);count+=n[ix].sum(axis=1)
    sample=total/count
    # Independent row-level fsum block construction with a fresh identical random stream.
    records={}
    for row in d.itertuples():
        records.setdefault((row.farm,int(row.day)),[]).append((row.y-row.candidate)**2-(row.y-row.baseline)**2)
    rng=np.random.default_rng(20261003+seed)
    ss=np.zeros(20000);nn=np.zeros(20000)
    for farm in ['F13','F47']:
        days=sorted(day for f,day in records if f==farm)
        blocks=[[x for day in days[i:i+5] for x in records[farm,day]] for i in range(0,len(days),5)]
        a=np.array([math.fsum(b) for b in blocks]);n=np.array([len(b) for b in blocks])
        ix=rng.integers(len(a),size=(20000,len(a)))
        ss+=a[ix].sum(axis=1);nn+=n[ix].sum(axis=1)
    compare(sample,ss/nn)
    return dict(p_worse=float(np.mean(sample>=0)),ci_adjusted=np.quantile(sample,[ALPHA,1-ALPHA]).tolist(),
                ci95=np.quantile(sample,[.025,.975]).tolist(),draws=20000)

def rmse(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    value=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(y,p))/len(y))
    compare([value],[np.sqrt(np.mean((y-p)**2))])
    return value

def original_guard_module():
    assert sha(V_PATH)==V_SHA
    sp=importlib.util.spec_from_file_location('original_r3_guard',V_PATH)
    obj=importlib.util.module_from_spec(sp);sp.loader.exec_module(obj)
    obj.bootstrap_runtime(with_support=True)
    return obj

def attach_ops(lab):
    assert sha(DP1)==DP1_SHA
    tree=ast.parse(DP1.read_text(encoding='utf-8-sig'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in {'run_len','day_feats'}]
    assert len(nodes)==2
    ns=dict(np=np,pd=pd)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DP1),'exec'),ns)
    d=lab.copy()
    extra=pd.concat([ns['day_feats'](g) for _,g in d.groupby(['farm','day'],sort=True)]).reindex(d.index)
    assert np.isfinite(extra[NEW].to_numpy(float)).all()
    return d.join(extra[NEW])

def synthetic():
    q=pd.DataFrame(dict(farm=['F13']*24+['F47']*24,day=[1]*48,hour=list(range(24))*2))
    rng=np.random.default_rng(21)
    et,lgb,mlp,pfn=rng.normal(size=(4,48))
    raw=.8*(.6*et+.3*lgb+.1*mlp)+.2*pfn
    avg=pd.Series(raw).groupby([q.farm,q.day]).expanding().mean().reset_index(level=[0,1],drop=True).sort_index().to_numpy()
    result=np.clip(.5*raw+.5*avg,-.2,.4)
    error=compare(result,scalar_candidate(q,et,lgb,mlp,pfn,-.2,.4))
    p=H/f'synthetic_{Path(__file__).stem}.json'
    with p.open('x',encoding='utf-8') as handle:
        json.dump(dict(status='PASS_SYNTHETIC_ARITHMETIC_ONLY',maxdiff=error,checks=CHECKS,fit=0,predict=0,real_score=0),handle,indent=2)
    print(p.name,error)

def verify():
    dest=[H/'full_verification_v1.json',H/'full_scores_v1.csv',H/'full_segments_v1.csv']
    assert all(not x.exists() for x in dest),'preserve existing full outputs'
    assert len(EXPECTED_RUN_SHA)==64 and len(EXPECTED_PREP_SHA)==64
    assert sha(H/EXPECTED_RUN_FILE)==EXPECTED_RUN_SHA
    assert sha(H/EXPECTED_PREP_FILE)==EXPECTED_PREP_SHA
    prepared=readj(H/EXPECTED_PREP_FILE)
    fit=readj(H/'fit_audit_v1.json')
    first_path=H/'first_fold_verification_v1.json'
    first=readj(first_path)
    assert prepared['status']=='PASS' and prepared['fit_count']==0
    assert fit['status']=='PASS' and fit['family']==FAMILY and len(fit['cells'])==66
    assert first['status']=='PASS'
    V=original_guard_module();S=V.S
    lab,core,wv,folds,outer=S.loadec()
    assert len(lab)==8640 and lab.row_id.is_unique and len(folds)==22
    assert sha(Path(core.__file__))==V.EXPECTED_DEPS['core']
    lab=attach_ops(lab)
    labels=lab.set_index('row_id').sub_ec
    base_cols=[c for c in core.BASE if c!='day']+['season']
    cols=base_cols+NEW
    assert len(base_cols)==14 and len(cols)==23 and 'day' not in cols
    expected={(v,k,s) for v,k,_,_ in folds for s in SEEDS}
    assert {p.name for p in OUT.glob('*_pred.csv')}=={f'{v}_{k}_{s}_pred.csv' for v,k,s in expected}
    assert {p.name for p in OUT.glob('*.json')}=={f'{v}_{k}_{s}.json' for v,k,s in expected}
    frames=[];guards=[];scalar_error=0.
    for v,k,tm,vm in folds:
        tr,q=S.seasonal(lab[tm],lab[vm],wv)
        tr,q=tr.reset_index(drop=True),q.reset_index(drop=True)
        train_keys=set(zip(tr.farm,tr.day));query_keys=set(zip(q.farm,q.day))
        assert not train_keys&query_keys
        assert all((farm,day+j) not in train_keys for farm,day in query_keys for j in [-1,0,1])
        old=dict(np.load(BASE/f'{v}_{k}_baseline.npz',allow_pickle=False))
        assert np.array_equal(old['row_id'],q.row_id)
        compare([old['lo'],old['hi']],[tr.sub_ec.min(),tr.sub_ec.max()])
        pfn=[]
        for c in [1,2,3,4]:
            p=dict(np.load(OLD/f'{v}_{k}_pfn_{c}.npz',allow_pickle=False))
            assert np.array_equal(p['row_id'],q.row_id)
            assert np.array_equal(p['context_row_id'],old[f'context_{c}']) and set(p['context_row_id'])<=set(tr.row_id)
            compare(p['sub_ec'],q.sub_ec);pfn.append(p['raw_pfn'])
        compare(old['old_pfn_raw'],np.mean(pfn,axis=0))
        for seed in SEEDS:
            r3_path=OLD/f'{v}_{k}_r3_{seed}.npz'
            r3=dict(np.load(r3_path,allow_pickle=False));r3_meta=readj(r3_path.with_suffix('.json'))
            guard=V.guard_original_r3(r3,r3_meta,sha(r3_path),tr,q,seed,v,k,labels)
            guards.append(guard)
            csv_path=OUT/f'{v}_{k}_{seed}_pred.csv'
            meta=readj(OUT/f'{v}_{k}_{seed}.json')
            assert meta['status']=='PASS' and meta['csv_sha256']==sha(csv_path)
            d=pd.read_csv(csv_path,float_precision='round_trip')
            assert d.row_id.is_unique and np.array_equal(d.row_id,q.row_id)
            assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==seed).all()
            for c in ['farm','day','hour']:assert np.array_equal(d[c],q[c])
            compare(d.y,q.sub_ec);compare(d.y,labels.reindex(d.row_id))
            baseline=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(q.row_id)
            compare(d.baseline,baseline)
            compare(d.baseline,np.clip(core.shrink(.8*r3['raw_r3']+.2*old['old_pfn_raw'],q),old['lo'],old['hi']))
            raw=.8*(.6*r3['raw_et']+.3*d.new_lgb_raw.to_numpy()+.1*r3['raw_mlp'])+.2*old['old_pfn_raw']
            compare(d.candidate,np.clip(core.shrink(raw,q),old['lo'],old['hi']))
            scalar_error=max(scalar_error,compare(d.candidate,scalar_candidate(q,r3['raw_et'],d.new_lgb_raw.to_numpy(),r3['raw_mlp'],old['old_pfn_raw'],old['lo'],old['hi'])))
            old_raw=.8*r3['raw_r3']+.2*old['old_pfn_raw']
            compare(raw-old_raw,.24*(d.new_lgb_raw.to_numpy()-r3['raw_lgb']))
            if (v,k,seed)==('DIAG10',0,7):assert meta['first_audit_sha256']==sha(first_path)
            frames.append(d)
    all_rows=pd.concat(frames,ignore_index=True)
    assert len(all_rows)==83160 and not all_rows.duplicated(['validator','fold','seed','row_id']).any()
    scores=[];segments=[];boots={}
    for (v,seed),d in all_rows.groupby(['validator','seed']):
        rb,rc=rmse(d.y,d.baseline),rmse(d.y,d.candidate)
        scores.append(dict(validator=v,seed=int(seed),n=len(d),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1)))
    for seed in SEEDS:
        d=all_rows[(all_rows.validator=='DIAG10')&(all_rows.seed==seed)]
        assert len(d)==8640
        high=d.groupby(['farm','day']).y.transform('mean')>=1
        assert high.sum()==744
        boots[seed]=boot(d,seed)
        for name,mask in [('high',high),('ordinary',~high),('late',d.day>=179),('F13',d.farm=='F13'),('F47',d.farm=='F47'),('hour0',d.hour==0)]:
            g=d[mask]
            segments.append(dict(seed=seed,segment=name,n=len(g),days=len(g[['farm','day']].drop_duplicates()),
                                 baseline=rmse(g.y,g.baseline),candidate=rmse(g.y,g.candidate),
                                 baseline_bias=math.fsum(g.baseline-g.y)/len(g),candidate_bias=math.fsum(g.candidate-g.y)/len(g)))
    sc=pd.DataFrame(scores)
    assert len(sc)==15
    passed=bool((sc.change_pct<0).all() and all(b['p_worse']<ALPHA and b['ci_adjusted'][1]<0 for b in boots.values()))
    sc.to_csv(dest[1],index=False,mode='x');pd.DataFrame(segments).to_csv(dest[2],index=False,mode='x')
    result=dict(status='PASS_ARITHMETIC',decision='PUBLIC_PASS_PENDING_REVIEW' if passed else 'REJECT',public_pass=passed,
                family=FAMILY,alpha=ALPHA,rows=len(all_rows),cells=66,checks=CHECKS+V.CHECKS,
                bootstrap=boots,scalar_maxdiff=scalar_error,r3_cache_audit=guards,
                runner_sha256=EXPECTED_RUN_SHA,preparation_sha256=EXPECTED_PREP_SHA,verifier_sha256=sha(Path(__file__)),
                fit=0,predict=0,test_reads=0,raw_ec_label_reads=0,EL1_rescore=0)
    with dest[0].open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(sc.to_string(index=False));print(result['decision'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--synthetic',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();assert args.synthetic != args.verify
    synthetic() if args.synthetic else verify()
