"""Independent preparation/PFN provenance replay; no fitting or performance scoring."""
from pathlib import Path
import sys, json, hashlib, importlib.util, csv
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
prep=json.loads((H/'preparation_v4.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import numpy as np,pandas as pd
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
season=load('season',prep['imported_season_path'])
sys.path.insert(0,str(ROOT/'집/클로드/submission14_ec_sg2'))
M=load('critic_readonly_current14',ROOT/'집/클로드/submission14_ec_sg2/model.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fhash=lambda frame,cols:hashlib.sha256(pd.util.hash_pandas_object(frame.loc[:,cols],index=False).to_numpy().tobytes()).hexdigest()
assert sha(Path(env.DATA)/'train_X.csv')=='21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300'
labels={}
with (ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['validator']!='DIAG10':continue
        i=r['row_id'];v=float(r['y'])
        if i in labels:assert labels[i]==v
        else:labels[i]=v
assert len(labels)==8640
raw=pd.read_csv(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
features=M.features(raw[['row_id']+M.RAW]);features=features[features.row_id.isin(labels)].copy()
features['sub_ec']=features.row_id.map(labels)
assert len(features)==8640
lookup=features.set_index('row_id');vectors=season.vectors(raw)
C=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/components'
records=[]
for rec in prep['records']:
    k=rec['k']
    with np.load(C/f'DIAG10_{k}_r3_7.npz',allow_pickle=False) as z:
        assert z['train_row_id'].astype(str).tolist()==rec['train_ids']
        assert z['row_id'].astype(str).tolist()==rec['query_ids']
    t=lookup.loc[rec['train_ids']].reset_index();q=lookup.loc[rec['query_ids']].reset_index()
    td=t[['farm','day']].drop_duplicates();qd=q[['farm','day']].drop_duplicates().reset_index(drop=True)
    ts,qs,_=season.mapping(td,qd,vectors);queryseason=dict(zip(qd.itertuples(index=False,name=None),qs))
    t['season']=[ts[(f,int(d))] for f,d in t[['farm','day']].itertuples(index=False,name=None)]
    q['season']=[queryseason[(f,int(d))] for f,d in q[['farm','day']].itertuples(index=False,name=None)]
    for part,frame in [('train',t),('query',q)]:
        for featurepart,cols in [('full',M.FULL_R3),('base',M.BASE_R3)]:
            assert fhash(frame,['row_id','sub_ec']+cols)==rec[f'{part}_{featurepart}_hash']
    th=fhash(t,['row_id','sub_ec']+M.FULL);qh=fhash(q,['row_id','sub_ec']+M.FULL)
    oldmeta=json.loads((C/f'DIAG10_{k}_r3_7.json').read_text(encoding='utf-8'))['provenance']
    assert (th,qh)==(oldmeta['train_hash'],oldmeta['validation_hash'])
    assert rec['bounds']==[float(t.sub_ec.min()),float(t.sub_ec.max())]
    assert M.FULL==prep['features']['FULL_PFN'] and M.FULL_R3==prep['features']['FULL_R3'] and M.BASE_R3==prep['features']['BASE_R3']
    for context in rec['PFN_sources']:
        seed=context['seed'];p=C/f'DIAG10_{k}_pfn_{seed}.npz';meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));prov=meta['provenance'];e=prov['environment']
        assert sha(p)==context['sha']==meta['prediction_sha256']
        assert (th,qh)==(prov['train_hash'],prov['validation_hash'])
        assert (len(t),len(q))==(prov['train_rows'],prov['validation_rows'])
        assert meta['seed']==seed and meta['fold']==['DIAG10',k]
        assert e['n_estimators']==4 and e['context_size']==2000 and e['inference_precision']=='float32'
        assert e['checkpoint_sha256']=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
        with np.load(p,allow_pickle=False) as z:
            ix=np.random.default_rng(seed).choice(len(t),size=2000,replace=False)
            assert np.array_equal(z['context_index'],ix)
            assert z['context_row_id'].astype(str).tolist()==t.row_id.iloc[ix].tolist()
            assert z['row_id'].astype(str).tolist()==q.row_id.tolist()
            assert z['raw_pfn'].shape==(len(q),) and np.isfinite(z['raw_pfn']).all()
        records.append(dict(fold=k,context_seed=seed,train_hash=th,query_hash=qh,npz_sha=sha(p),context_rows=2000,train_rows=len(t),query_rows=len(q)))
        print('PASS_PFN_PROVENANCE',k,seed,flush=True)
out=dict(status='PASS_FULL_FEATURE_HASH_AND_40_PFN_CONTEXTS',fit=0,performance_scores=0,folds=10,caches=len(records),FULL_R3=47,BASE_R3=23,FULL_PFN=38,records=records,method='Independent csv label reader, frame ordering/hash calculation, direct season.mapping replay; current immutable features function reused. No PFN inference replay.',ablation_bias_definition='raw ET 24h mean minus truth 24h mean; smoothing auxiliary only')
assert len(records)==40
with (H/'critic_verify_pfn_provenance_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(out['status'],flush=True)
