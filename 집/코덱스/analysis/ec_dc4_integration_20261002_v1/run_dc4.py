"""사전 고정 DC4 독립 재현 및 전체 v2 통합. 원본 파일 읽기 전용."""
from pathlib import Path
import sys, os, importlib.util, argparse, json, gc, types, contextlib, io
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import env_extra
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
SEEDS=[7,101,2024]
W=['out_temp','out_hum','out_rad','out_wspd']

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

common=load('dc4_readonly_common',HERE.parent/'ec_model_common_20261002_v1/common.py')
engine=load('dc4_readonly_engine',HERE.parent/'ec_stage2_tabpfn_20261002_v2/run_stage2_tabpfn.py')
source=load('dc4_readonly_source',ROOT/'집/클로드/research/ec2_DC4_exact_twin_anchor_v1.py')
engine.OUT=OUT
engine.R3_SEEDS=SEEDS
engine.PFN_SEEDS=[1,2,3,4]
common.K=2;common.ALPHA=.0125

def pav(values):
    # Independent weighted pool-adjacent-violators; no sklearn isotonic call.
    blocks=[]
    for k,y in enumerate(values):
        blocks.append([k,k+1,float(y),1])
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    result=np.empty(len(values),float)
    for a,b,s,n in blocks:result[a:b]=s/n
    return result

def vectors(raw):
    a=common.core.identify(raw)
    answer={}
    for key,g in a.groupby(['farm','day'],sort=True):
        assert len(g)==24 and set(g.hour)==set(range(24))
        answer[key]=g.sort_values('hour')[W].to_numpy(float).T
    return answer

def mapping(tdays,qdays,vectors):
    p1=[(f,int(d)) for f,d in tdays.itertuples(index=False,name=None) if d<179]
    reference=np.stack([vectors[k] for k in p1])
    mu=np.nanmean(reference,axis=(0,2));sd=np.nanstd(reference,axis=(0,2));sd[sd==0]=1
    ref=(reference-mu[None,:,None])/sd[None,:,None]
    seasons={key:float(key[1]) for key in p1};fits={};notes={}
    for farm in ['F13','F47']:
        days=sorted(int(d) for f,d in tdays.itertuples(index=False,name=None) if f==farm and d>=179)
        x,y=[],[];nearest=[]
        for d in days:
            query=(vectors[(farm,d)]-mu[:,None])/sd[:,None]
            distance=np.sqrt(np.nanmean((ref-query)**2,axis=(1,2)))
            nearest.append(float(p1[int(np.nanargmin(distance))][1]))
            selected=np.flatnonzero(distance<=.05)
            if len(selected):x.append(d);y.append(float(np.mean([p1[i][1] for i in selected])))
        nanchors=len(x);fallback=nanchors<2
        if fallback:x,y=days,nearest
        assert len(x)>0, '학습 2차 기준점 없음'
        x=np.asarray(x,float);fitted=pav(y)
        fits[farm]=(x,fitted)
        for d in days:seasons[(farm,d)]=float(np.interp(d,x,fitted))
        notes[farm]={'exact_anchors':nanchors,'training_late_days':len(days),'fallback':fallback}
    out=[]
    for farm,d in qdays.itertuples(index=False,name=None):
        if d<179:
            x=sorted(dd for f,dd in p1 if f==farm);out.append(float(np.interp(d,x,x)))
        else:out.append(float(np.interp(d,*fits[farm])))
    return seasons,np.asarray(out),notes

def transform(raw,tr,va,check=False):
    td=tr[['farm','day']].drop_duplicates();qd=va[['farm','day']].drop_duplicates().reset_index(drop=True)
    v=vectors(raw);ts,qs,notes=mapping(td,qd,v)
    if check:
        with contextlib.redirect_stdout(io.StringIO()):
            a,b=source.season_index(td,qd,source.weather_vectors(common.core.identify(raw)))
        gap=max([abs(ts[k]-a[k]) for k in ts]+[float(np.max(np.abs(qs-b)))])
        assert gap<=1e-10,(gap,notes)
        allowed=set(td.itertuples(index=False,name=None))
        changed={k:(value.copy() if k in allowed else value*19+999) for k,value in v.items()}
        aa,bb,_=mapping(td,qd,changed)
        assert ts==aa and np.array_equal(qs,bb)
        sh,zz,_=mapping(td,qd.iloc[::-1].reset_index(drop=True),v)
        assert ts==sh and np.array_equal(qs,zz[::-1])
        notes['checks']={'source_max_difference':gap,'query_and_future_weather_invariance':True,'query_order_invariance':True}
    tr=tr.copy();va=va.copy();qm=dict(zip(qd.itertuples(index=False,name=None),qs))
    tr['season']=[ts[(f,int(d))] for f,d in zip(tr.farm,tr.day)]
    va['season']=[qm[(f,int(d))] for f,d in zip(va.farm,va.day)]
    return tr,va,notes

def seasonal_core(core):
    return types.SimpleNamespace(FULL=[c for c in core.FULL if c!='day']+['season'],
                                 BASE=[c for c in core.BASE if c!='day']+['season'],
                                 et=core.et,lg=core.lg,shrink=core.shrink)

def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,lab,folds,locks,core=common.prepare()
    base={'code':engine.sha(__file__),'protocol':engine.sha(HERE/'PROTOCOL.md'),
          'source':engine.sha(source.__file__),'engine':engine.sha(engine.__file__),
          'shared':common.manifest(),'environment':engine.environment_manifest(),
          'seeds':SEEDS,'pfn_contexts':[1,2,3,4],'feature_order':'remove day; append season'}
    return raw,lab,folds,locks,core,base

def run(component,shard):
    raw,lab,folds,locks,core,base=setup();sc=seasonal_core(core)
    log=OUT/f'shard{shard}_{component}.log'
    for ordinal,fold in enumerate(folds):
        if ordinal%3!=shard:continue
        tr,va=common.split_fold(raw,lab,fold,locks)
        tr,va,notes=transform(raw,tr,va,check=True)
        prov=engine.fold_provenance(tr,va,sc,base)|{'season_checks':notes}
        with engine.fold_lock(fold):
            if component=='r3':
                for seed in SEEDS:engine.fit_r3(tr,va,sc,fold,seed,prov,log)
            else:
                for seed in [1,2,3,4]:engine.fit_pfn(tr,va,sc,fold,seed,prov,log)
        engine.log(f'{fold[0]}/{fold[1]} {component} 완료',log)
    engine.atomic_json(OUT/f'shard{shard}_{component}_complete.json',{'status':'COMPLETE','base':base})

def collect(et_only):
    raw,lab,folds,locks,core,base=setup();sc=seasonal_core(core)
    original=pd.read_csv(ROOT/'집/클로드/research/local/ec2_DC4_oof.csv',float_precision='round_trip',
                         usecols=['row_id','sub_ec','season','validator','fold']+[f'{kind}_{s}' for kind in ['base','seas'] for s in SEEDS])
    frames=[];checks=[]
    for fold in folds:
        tr,va=common.split_fold(raw,lab,fold,locks);tr,va,notes=transform(raw,tr,va,check=True)
        prov=engine.fold_provenance(tr,va,sc,base)|{'season_checks':notes}
        ov=original[original.validator.eq(fold[0])&original.fold.eq(fold[1])].set_index('row_id').loc[va.row_id]
        assert np.array_equal(ov.sub_ec.to_numpy(),va.sub_ec.to_numpy())
        assert np.max(np.abs(ov.season.to_numpy()-va.season.to_numpy()))<=1e-10
        bags=[]
        if not et_only:
            r3,pfn=engine.verify_fold(tr,va,sc,fold,prov)
            bag=np.mean([pfn[s]['raw_pfn'] for s in [1,2,3,4]],axis=0)
        for seed in SEEDS:
            if et_only:
                key=engine.canonical_hash(prov|{'kind':'R3 original .6ET+.3LGB+.1MLP','seed':seed})
                z,_=engine.load_cache(engine.cache_path(fold,'r3',seed),key,va.row_id)
            else:z=r3[seed]
            season_et=common.finish(z['raw_et'],tr,va)
            etgap=float(np.max(np.abs(season_et-ov[f'seas_{seed}'].to_numpy())))
            assert etgap<=1e-8,('ET 독립 재현',fold[:2],seed,etgap)
            daypath=ROOT/'집/코덱스/local/ec_member_ablation_20261002_v1'/f'{fold[0]}_{fold[1]}_seed{seed}_members.npz'
            daymeta=json.loads(daypath.with_suffix('.json').read_text(encoding='utf-8'))
            assert engine.sha(daypath)==daymeta['npz_sha256']
            with np.load(daypath) as old:
                assert old['row_id'].tolist()==va.row_id.tolist()
                dayet=common.finish(old['et'],tr,va)
            assert np.max(np.abs(dayet-ov[f'base_{seed}'].to_numpy()))<=1e-8
            checks.append({'validator':fold[0],'fold':fold[1],'seed':seed,'ET_max_gap':etgap,'season_checks':notes})
            f=va[['row_id','farm','day','hour','block','sub_ec','season']].copy()
            f['validator'],f['validation_fold'],f['seed']=fold[0],fold[1],seed
            if et_only:f['v2']=ov[f'base_{seed}'].to_numpy();f['season_et']=season_et
            else:
                ref=common.baseline(va,fold[0],fold[1],seed)
                f['v2']=ref['v2'];f['season_v2']=common.finish(.8*z['raw_r3']+.2*bag,tr,va)
                f['season_r3']=common.finish(z['raw_r3'],tr,va);f['season_pfn']=common.finish(bag,tr,va)
                f['day_r3']=ref['r3'];f['day_pfn']=ref['pfn_finished']
            frames.append(f)
    data=pd.concat(frames,ignore_index=True);arm='season_et' if et_only else 'season_v2'
    result=common.evaluate(data,[arm]);result['independent_reproduction']=checks
    result['independent_verification_pending']=True;result['final_lock_scored']=False
    if et_only:
        result['replication_pass']=True;result['not_a_new_candidate']=True
        for d in result['decisions']:d['adopted']=False;d['label']='기존 ET 발견 재현 확인'
    else:
        diagnostics=[]
        for (farm,late),g in data[data.validator.eq('DIAG10')].assign(late=lambda f:f.day.ge(179)).groupby(['farm','late']):
            av=g.groupby('row_id')[['sub_ec','v2',arm,'day_r3','season_r3','day_pfn','season_pfn']].mean()
            diagnostics.append({'farm':farm,'late':bool(late),'n':len(av),**{c:common.rmse(av.sub_ec,av[c]) for c in ['v2',arm,'day_r3','season_r3','day_pfn','season_pfn']}})
        result['diagnostics']=diagnostics
    tag='et_replication' if et_only else 'v2_integration'
    engine.atomic_csv(OUT/f'{tag}_oof.csv',data)
    engine.atomic_json(HERE/f'{tag}_result.json',result)
    engine.atomic_csv(HERE/f'{tag}_scores.csv',pd.DataFrame(result['scores']))
    engine.atomic_csv(HERE/f'{tag}_bootstrap.csv',pd.DataFrame(result['bootstrap']))
    print(json.dumps({'tag':tag,'ensemble':result['ensemble'],'decisions':result['decisions']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--component',choices=['r3','pfn']);p.add_argument('--shard',type=int,choices=[0,1,2]);p.add_argument('--collect',choices=['et','v2'])
    a=p.parse_args()
    if a.collect:collect(a.collect=='et')
    else:
        assert a.component and a.shard is not None
        run(a.component,a.shard)
