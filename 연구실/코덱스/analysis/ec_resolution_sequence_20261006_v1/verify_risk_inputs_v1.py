"""Audit partial nested risk inputs only: no risk fitting, no stage1 completion audit."""
from pathlib import Path
import sys,json,csv,math,hashlib,datetime
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(H));import verify_stage1_v1 as B
np,pd,M=B.np,B.pd,B.M;L=ROOT/'연구실/코덱스/local'/H.name;C=L/'nested_components';O=L/'risk_inputs_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ar(a):return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11),(a,b)
def avg(a):return math.fsum(map(float,a))/len(a)
def daykey(r):return (str(r)[:3],int(str(r)[4:7]))
def smooth(values,q):
    out=[];running={}
    for v,r in zip(values,q.itertuples()):key=(r.farm,int(r.day));running.setdefault(key,[]).append(float(v));out.append(.5*float(v)+.5*avg(running[key]))
    return out
def ordered_hours(frame):
    for key,q in frame.groupby(['farm','day'],sort=False):assert q.hour.tolist()==list(range(24)),(key,'prefix is not ordered 0..23')
def cache(path,e,kind,seed):
    meta=js(path.with_suffix('.json'));assert meta['sha']==sha(path);assert meta['signature']['record_sha']==hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest();assert meta['signature']['prepared_sha']==prepared_sha;assert meta['signature']['seed']==seed
    with np.load(path,allow_pickle=False) as z:r={k:z[k].copy() for k in z.files}
    assert r['row_id'].tolist()==e['query_ids'] and r['train_row_id'].tolist()==e['train_ids'];assert all(np.isfinite(v).all() for v in r.values() if v.dtype.kind in 'fc')
    return r
prep_path=ROOT/'집/코덱스/analysis/ec_actual_A_nested_oof_20261006_v1/preparation_v4.json';prep=js(prep_path);prepared_sha=sha(prep_path);manifest=js(H/'risk_input_preparation_v1.json');assert manifest['fit']==0 and not manifest['full80'];assert manifest['source_sha']==sha(H/'assemble_risk_inputs_v1.py')
z=pd.read_csv(ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv',float_precision='round_trip');z=z[z.validator=='DIAG10'];y=z[['row_id','y']].drop_duplicates();assert len(y)==8640
raw=pd.read_csv(Path(B.env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True);f=M.features(raw[['row_id']+M.RAW]).merge(y.rename(columns={'y':'sub_ec'}),on='row_id',how='inner',validate='one_to_one');vec=M.vectors(raw)
records={};audited_caches=0;splitchecks=[]
for k in range(4):
    es=[e for e in prep['records'] if e['v']=='DIAG10' and e['k']==k];assert len(es)==4
    outerids=es[0]['outer_train_ids'];outerquery=es[0]['outer_query_ids'];assert len(outerids)==len(set(outerids));assert not set(outerids)&set(outerquery)
    originalC=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/components'
    with np.load(originalC/f'DIAG10_{k}_r3_7.npz',allow_pickle=False) as v:assert v['train_row_id'].astype(str).tolist()==outerids and v['row_id'].astype(str).tolist()==outerquery
    seen=[];outerban={(farm,day+j) for farm,day in map(daykey,outerquery) for j in [-1,0,1]}
    for e in es:
        assert e['outer_train_ids']==outerids and e['outer_query_ids']==outerquery;assert set(e['train_ids'])<=set(outerids) and set(e['query_ids'])<=set(outerids);assert not set(e['train_ids'])&set(e['query_ids'])
        innerban={(farm,day+j) for farm,day in map(daykey,e['query_ids']) for j in [-1,0,1]};assert not set(map(daykey,e['train_ids']))&(outerban|innerban);seen+=e['query_ids']
        t,q=B.transform(B.order(f,e['train_ids']),B.order(f,e['query_ids']),vec);ordered_hours(q)
        assert ar(t[M.FULL].to_numpy())==e['train_full_sha'] and ar(q[M.FULL].to_numpy())==e['query_full_sha'];assert ar(t[M.BASE].to_numpy())==e['train_base_sha'] and ar(q[M.BASE].to_numpy())==e['query_base_sha'];assert ar(t.sub_ec.to_numpy())==e['train_target_sha'] and ar(q.sub_ec.to_numpy())==e['query_target_sha'];close(t.sub_ec.min(),e['bounds'][0]);close(t.sub_ec.max(),e['bounds'][1])
        components={}
        for seed in [7,101,2024]:
            r=cache(C/f'DIAG10_{k}_{e["j"]}_r3_{seed}.npz',e,'r3',seed);audited_caches+=1
            for i in range(len(q)):close(r['raw'][i],math.fsum([.6*r['et'][i],.3*r['lgb'][i],.1*r['mlp'][i]]))
            components[seed]=r
        ps=[]
        for seed in [1,2,3,4]:
            r=cache(C/f'DIAG10_{k}_{e["j"]}_pfn_{seed}.npz',e,'pfn',seed);audited_caches+=1;ix=np.random.default_rng(seed).choice(len(t),min(2000,len(t)),replace=False);assert r['context_row_id'].tolist()==t.row_id.iloc[ix].tolist();ps.append(r['raw'])
        bag=[avg([p[i] for p in ps]) for i in range(len(q))];records[k,e['j']]=(t,q,components,bag,e)
    assert len(seen)==len(set(seen))==len(outerids) and set(seen)==set(outerids);splitchecks.append(dict(outer_fold=k,inner_jobs=4,outer_train_rows=len(outerids),outer_query_rows=len(outerquery),query_coverage='exactly once',outer_and_inner_adjacent_day_purge=True))
files=[]
for item in manifest['files']:
    k,seed=item['k'],item['seed'];trainpath=O/f'train_{k}_{seed}.csv';querypath=O/f'query_{k}_{seed}.csv';assert sha(trainpath)==item['train_sha'] and sha(querypath)==item['query_sha']
    train=pd.read_csv(trainpath,float_precision='round_trip');query=pd.read_csv(querypath,float_precision='round_trip');assert len(train)==item['train_rows'] and len(query)==item['query_rows'];ordered_hours(train);ordered_hours(query);assert not set(train.row_id)&set(query.row_id)
    es=[e for e in prep['records'] if e['v']=='DIAG10' and e['k']==k];assert train.row_id.tolist()==es[0]['outer_train_ids'];assert query.row_id.tolist()==es[0]['outer_query_ids']
    byid=train.set_index('row_id')
    for e in es:
        t,q,components,bag,e=records[k,e['j']];r=components[seed];actual=byid.loc[e['query_ids']].reset_index();pd.testing.assert_frame_equal(actual[M.FULL],q[M.FULL],check_dtype=False,atol=1e-12,rtol=1e-12);assert (actual.inner_j==e['j']).all()
        mix=[math.fsum([.48*r['et'][i],.24*r['lgb'][i],.08*r['mlp'][i],.2*bag[i]]) for i in range(len(q))];smix=smooth(mix,q)
        for n,v in [('et',r['et']),('lgb',r['lgb']),('mlp',r['mlp']),('pfn',bag)]:
            sm=smooth(v,q)
            for i in range(len(q)):close(actual.iloc[i]['raw_'+n],v[i]);close(actual.iloc[i]['smooth_'+n],sm[i])
        for i in range(len(q)):close(actual.A.iloc[i],max(e['bounds'][0],min(e['bounds'][1],smix[i])));close(actual.sub_ec.iloc[i],q.sub_ec.iloc[i])
    expected=pd.read_csv(L/'nested_snapshot'/f'OOF_DIAG10_{k}_{seed}.csv',float_precision='round_trip');assert expected.row_id.tolist()==train.row_id.tolist()
    for a,b in zip(expected.A,train.A):close(a,b)
    cq=z[(z.fold==k)&(z.seed==seed)].set_index('row_id').loc[query.row_id].reset_index();outert,outerq=B.transform(B.order(f,es[0]['outer_train_ids']),B.order(f,es[0]['outer_query_ids']),vec);pd.testing.assert_frame_equal(query[M.FULL],outerq[M.FULL],check_dtype=False,atol=1e-12,rtol=1e-12)
    for n,col in [('et','raw_et'),('lgb','raw_lgb'),('mlp','raw_mlp'),('pfn','old_pfn_raw')]:
        v=cq[col].to_numpy();sm=smooth(v,query)
        for i in range(len(query)):close(query.iloc[i]['raw_'+n],v[i]);close(query.iloc[i]['smooth_'+n],sm[i])
    mixture=[math.fsum([.48*cq.raw_et.iloc[i],.24*cq.raw_lgb.iloc[i],.08*cq.raw_mlp.iloc[i],.2*cq.old_pfn_raw.iloc[i]]) for i in range(len(query))];sm=smooth(mixture,query)
    for i in range(len(query)):close(query.A.iloc[i],cq.baseline.iloc[i]);close(query.A.iloc[i],max(float(outert.sub_ec.min()),min(float(outert.sub_ec.max()),sm[i])));close(query.sub_ec.iloc[i],cq.y.iloc[i])
    for frame in [train,query]:
        for _,q in frame.groupby(['farm','day'],sort=False):
            vals=[]
            for row in q.itertuples():vals.append(row.A);close(row.prefix_A,avg(vals))
    files.append(dict(k=k,seed=seed,train_rows=len(train),query_rows=len(query)))
assert len(files)==12 and audited_caches==112
out=dict(status='PASS_PARTIAL_INPUT_AUDIT_ONLY',risk_model_fit=0,partial_outer_folds=[0,1,2,3],full80=False,train_query_pairs=len(files),audited_component_cache_pairs=audited_caches,splitchecks=splitchecks,files=files,feature_boundary='sub_ec is label/evaluation only; inner_j is split metadata; explicit causal feature whitelist required before risk fitting',limitations=['source dataset assembly was audited, not a trained risk model','12 seed-fold outputs share4 outer splits','inner OOF training scores and outer-fit query scores have different model training sizes/distributions'])
path=H/('verify_risk_inputs_v1_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=out['status'],train_query_pairs=len(files),component_caches=audited_caches,output=str(path)),ensure_ascii=False,indent=2))
