"""Read-only stage1 cache/output audit. Never fit or import the worker."""
from pathlib import Path
import sys,json,csv,math,hashlib,importlib.util,argparse,datetime
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
SOURCE=ROOT/'집/코덱스/analysis/ec_submission10_season_20261002_v1/model.py'
spec=importlib.util.spec_from_file_location('audit_readonly_season_source',SOURCE);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
O=ROOT/'연구실/코덱스/local'/H.name/'stage1_v2';C=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/components'
TARGETS={('F13',98),('F13',112),('F47',160),('F47',161)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fh(f):return hashlib.sha256(pd.util.hash_pandas_object(f[['row_id','sub_ec']+M.FULL],index=False).values.tobytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def same(a,b,tol=1e-10):assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)
def mean(v):return math.fsum(map(float,v))/len(v)
def order(f,ids):return f.set_index('row_id').loc[list(ids)].reset_index()
def transform(t,q,vec):
    td=t[['farm','day']].drop_duplicates();qd=q[['farm','day']].drop_duplicates().reset_index(drop=True);ts,qs,_=M.mapping(td,qd,vec);qm=dict(zip(qd.itertuples(index=False,name=None),qs));t=t.copy();q=q.copy()
    t['season']=[ts[(r.farm,int(r.day))] for r in t.itertuples()];q['season']=[qm[(r.farm,int(r.day))] for r in q.itertuples()];return t,q
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--complete',action='store_true');a=parser.parse_args()
    prep=js(H/'preparation_v2.json')
    for p,digest in prep['sha'].items():assert sha(p)==digest,(p,'source/input changed')
    assert sha(ROOT/'연구실/코덱스/local'/H.name/'tabpfn-v2-regressor.ckpt')==prep['checkpoint']
    op=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv';z=pd.read_csv(op,float_precision='round_trip');z=z[z.validator=='DIAG10'];y=z[['row_id','y']].drop_duplicates();assert len(y)==8640
    raw=pd.read_csv(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True);f=M.features(raw[['row_id']+M.RAW]).merge(y.rename(columns={'y':'sub_ec'}),on='row_id',how='inner',validate='one_to_one');vec=M.vectors(raw)
    originals={}
    for k in [0,1,8]:
        with np.load(C/f'DIAG10_{k}_r3_7.npz',allow_pickle=False) as v:ti=v['train_row_id'].astype(str);qi=v['row_id'].astype(str)
        t,q=transform(order(f,ti),order(f,qi),vec);meta=js(C/f'DIAG10_{k}_r3_7.json')['provenance'];assert fh(t)==meta['train_hash'] and fh(q)==meta['validation_hash'];originals[k]=(t,q)
    target=f[[(r.farm,int(r.day)) in TARGETS for r in f.itertuples()]].sort_values(['farm','day','hour']).reset_index(drop=True);assert len(target)==96
    bans={(farm,day+j) for farm,day in TARGETS for j in [-1,0,1]};frames={}
    for k,(t,q) in originals.items():
        ids=t.loc[[(r.farm,int(r.day)) not in bans for r in t.itertuples()],'row_id'];frames[f'trainfold{k}_purged']=transform(order(f,ids),target,vec)
    common=set.intersection(*[set(t.row_id) for t,q in frames.values()]);ids=frames['trainfold0_purged'][0].row_id[frames['trainfold0_purged'][0].row_id.isin(common)];frames['common_intersection']=transform(order(f,ids),target,vec)
    for name,(t,q) in frames.items():
        assert fh(t)==prep['jobs'][name]['train_hash'] and fh(q)==prep['jobs'][name]['query_hash'];assert not set(t.row_id)&set(q.row_id)
        assert all((r.farm,int(r.day)) not in bans for r in t.itertuples());assert t[M.FULL].notna().any().all(),'imputer can drop columns: trace labeling unsupported'
    t,q=originals[0];q=q[[(r.farm,int(r.day)) in TARGETS for r in q.itertuples()]].reset_index(drop=True);assert len(q)==48;frames['replay_fold0']=(t,q)
    data={};checks=[];incomplete=[];replay=[]
    # Completed pairs only; in-flight staging or unsigned NPZ is never treated as PASS.
    for p in sorted(O.glob('*.npz')):
        if p.name.endswith('.staging.npz'):continue
        side=p.with_suffix('.json')
        if not side.exists():incomplete.append(p.name);continue
        meta=js(side);assert meta['sha']==sha(p)
        stem=p.stem;kind='r3' if '_r3_' in stem else 'pfn';name,seedtext=stem.rsplit('_'+kind+'_',1);seed=int(seedtext);t,q=frames[name]
        assert meta['key']==dict(preparation_sha=sha(H/'preparation_v2.json'),source_sha=sha(H/'stage1_v2.py'),train=fh(t),query=fh(q))
        with np.load(p,allow_pickle=False) as v:cache={k:v[k].copy() for k in v.files}
        assert np.array_equal(cache['row_id'].astype(str),q.row_id.to_numpy(str));assert np.array_equal(cache['train_row_id'].astype(str),t.row_id.to_numpy(str))
        assert all(np.isfinite(v).all() for v in cache.values() if v.dtype.kind in 'fc')
        if kind=='r3':
            for i in range(len(q)):same(cache['raw_r3'][i],math.fsum([.6*cache['raw_et'][i],.3*cache['raw_lgb'][i],.1*cache['raw_mlp'][i]]))
        else:
            ix=np.random.default_rng(seed).choice(len(t),min(2000,len(t)),replace=False);assert np.array_equal(cache['context_index'],ix);assert np.array_equal(cache['context_row_id'].astype(str),t.row_id.iloc[ix].to_numpy(str))
        if name=='replay_fold0':
            with np.load(C/f'DIAG10_0_{kind}_{seed}.npz',allow_pickle=False) as v:
                positions={str(r):i for i,r in enumerate(v['row_id'])};ix=[positions[r] for r in q.row_id];names=['raw_et','raw_lgb','raw_mlp','raw_r3'] if kind=='r3' else ['raw_pfn'];gap=max(abs(float(cache[n][i])-float(v[n][ix[i]])) for n in names for i in range(len(q)));assert gap<=(1e-8 if kind=='r3' else 1e-5)
                if kind=='pfn':assert np.array_equal(cache['context_index'],v['context_index'])
            replay.append(dict(kind=kind,seed=seed,maxdiff=gap,rows=len(q)))
        data[(name,kind,seed)]=cache;checks.append(stem)
    traces=[]
    for p in O.glob('*_et_paths.json'):
        name=p.name.removesuffix('_et_paths.json');t,q=frames[name];rows=js(p);assert len(rows)==1200;cache=data.get((name,'r3',7))
        if cache is None:continue
        med=t[M.FULL].median();index={(r.farm,int(r.day),int(r.hour)):i for i,r in enumerate(q.itertuples())}
        for farm,good,bad in [('F47',160,161),('F13',98,112)]:
            group=[r for r in rows if r['farm']==farm];assert len(group)==600 and {r['tree'] for r in group}==set(range(600))
            gi,bi=index[(farm,good,0)],index[(farm,bad,0)]
            for r in group:
                same(r['difference'],r['bad_prediction']-r['good_prediction']);assert all(math.isfinite(r[c]) for c in ['good_prediction','bad_prediction','difference'])
                s=r['first_divergence']
                if s:
                    c=s['feature'];assert c in M.FULL
                    for label,i in [('good',gi),('bad',bi)]:v=q.iloc[i][c];v=v if pd.notna(v) else med[c];same(s[label+'_value'],v)
                    assert (s['good_value']<=s['threshold'])!=(s['bad_value']<=s['threshold'])
            same(mean([r['good_prediction'] for r in group]),cache['raw_et'][gi]);same(mean([r['bad_prediction'] for r in group]),cache['raw_et'][bi]);same(mean([r['difference'] for r in group]),cache['raw_et'][bi]-cache['raw_et'][gi]);traces.append(dict(context=name,farm=farm,trees=600))
    scores=[]
    if a.complete:
        assert not (O/'worker.lock').exists(),'worker still active';receipt=js(H/'stage1_receipt_v2.json');dest=H/'stage1_rows_v2.csv';assert receipt['output_sha']==sha(dest)
        expected={('replay_fold0','r3',s) for s in [7,101,2024]}|{('replay_fold0','pfn',1)}
        for name in prep['jobs']:expected|={(name,'r3',s) for s in [7,101,2024]}|{(name,'pfn',s) for s in [1,2,3,4]}
        assert set(data)==expected and len(traces)==8 and not incomplete
        with dest.open(encoding='utf-8-sig',newline='') as stream:output=list(csv.DictReader(stream))
        assert len(output)==1152;lookup={(r['context'],int(r['seed']),r['row_id']):r for r in output};assert len(lookup)==len(output)
        for name in prep['jobs']:
            t,q=frames[name];perday={};lo=float(t.sub_ec.min());hi=float(t.sub_ec.max())
            for seed in [7,101,2024]:
                r3=data[(name,'r3',seed)];bag=[mean([data[(name,'pfn',s)]['raw_pfn'][i] for s in [1,2,3,4]]) for i in range(len(q))];mix=[.8*float(r3['raw_r3'][i])+.2*bag[i] for i in range(len(q))];running={}
                for i,row in enumerate(q.itertuples()):
                    key=(row.farm,int(row.day));running.setdefault(key,[]).append(mix[i]);prediction=max(lo,min(hi,.5*mix[i]+.5*mean(running[key])));r=lookup[(name,seed,row.row_id)]
                    for field,value in [('raw_pfn',bag[i]),('raw_mix',mix[i]),('prediction',prediction),('clip_lo',lo),('clip_hi',hi),('sub_ec',row.sub_ec),('season',row.season)]:same(float(r[field]),value)
                    for n in ['raw_et','raw_lgb','raw_mlp']:same(float(r[n]),r3[n][i])
                    perday.setdefault(key,{}).setdefault(row.row_id,dict(y=float(row.sub_ec),pred=[]))['pred'].append(prediction)
            for (farm,day),rr in perday.items():
                errors=[mean(r['pred'])-r['y'] for r in rr.values()];assert len(errors)==24;scores.append(dict(context=name,farm=farm,day=day,bias=mean(errors),rmse=math.sqrt(mean([e*e for e in errors]))))
        assert len(scores)==16
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f');out=dict(status='PASS_COMPLETE_DIAGNOSTIC' if a.complete else 'PASS_COMPLETED_CACHE_SNAPSHOT_ONLY',cache_pairs=len(checks),checked_caches=checks,incomplete_unsigned_npz=incomplete,replay=replay,traces=traces,daily_ensemble_scores=scores,limitations=['no fitting','not model adoption or utility validation','first split earlier-node correctness cannot be reconstructed without serialized full tree','replay covers heldout48 rows, not wholefold; float32PFN tolerance1e-5'])
    path=H/(f'verify_stage1_{"complete" if a.complete else "snapshot"}_{stamp}.json')
    with path.open('x',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=out['status'],cache_pairs=len(checks),replay=replay,trace_pairs=len(traces),daily_score_rows=len(scores),output=str(path)),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
