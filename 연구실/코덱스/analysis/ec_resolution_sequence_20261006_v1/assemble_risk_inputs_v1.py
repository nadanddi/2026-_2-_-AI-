from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(H));import stage1_v2 as S
import numpy as np,pandas as pd
L=S.L;C=L/'nested_components';O=L/'risk_inputs_v1';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((ROOT/'집/코덱스/analysis/ec_actual_A_nested_oof_20261006_v1/preparation_v4.json').read_text(encoding='utf-8'))
op=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv';z=pd.read_csv(op,float_precision='round_trip');z=z[z.validator=='DIAG10']
y=z[['row_id','y']].drop_duplicates();raw=pd.read_csv(Path(S.env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])]
f=S.M.features(raw[['row_id']+S.M.RAW]).merge(y.rename(columns={'y':'sub_ec'}),on='row_id',how='inner',validate='one_to_one');vec=S.M.vectors(raw)
manifest=[]
for k in range(4):
    es=[e for e in p['records'] if e['v']=='DIAG10' and e['k']==k];assert len(es)==4
    for seed in [7,101,2024]:
        assembled=[]
        for e in es:
            t=S.ordered(f,e['train_ids']);q=S.ordered(f,e['query_ids']);tt,qq,_=S.season(t,q,vec)
            # Declared original nested feature arrays are reproduced bit-for-bit.
            def ar(x):return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
            assert ar(tt[S.M.FULL].to_numpy())==e['train_full_sha']
            assert ar(qq[S.M.FULL].to_numpy())==e['query_full_sha']
            path=C/f"DIAG10_{k}_{e['j']}_r3_{seed}.npz";meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            assert sha(path)==meta['sha']
            assert meta['signature']['record_sha']==hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()
            with np.load(path,allow_pickle=False) as rr:
                assert rr['row_id'].tolist()==e['query_ids'] and rr['train_row_id'].tolist()==e['train_ids']
                et,lg,mlp=rr['et'].copy(),rr['lgb'].copy(),rr['mlp'].copy();assert np.max(abs(.6*et+.3*lg+.1*mlp-rr['raw']))<1e-12
            ps=[]
            for s in [1,2,3,4]:
                path=C/f"DIAG10_{k}_{e['j']}_pfn_{s}.npz";meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert sha(path)==meta['sha']
                assert meta['signature']['record_sha']==hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()
                with np.load(path,allow_pickle=False) as rr:
                    assert rr['row_id'].tolist()==e['query_ids'] and rr['train_row_id'].tolist()==e['train_ids']
                    ix=np.random.default_rng(s).choice(len(tt),min(2000,len(tt)),replace=False);assert rr['context_row_id'].tolist()==tt.row_id.iloc[ix].tolist();ps.append(rr['raw'].copy())
            bag=np.mean(ps,axis=0);mix=.48*et+.24*lg+.08*mlp+.2*bag
            frame=qq[['row_id','farm','day','hour','sub_ec']+S.M.FULL].copy();frame=frame.loc[:,~frame.columns.duplicated()]
            for n,arr in [('et',et),('lgb',lg),('mlp',mlp),('pfn',bag)]:frame['raw_'+n]=arr;frame['smooth_'+n]=S.M.shrink(arr,qq)
            frame['A']=np.clip(S.M.shrink(mix,qq),*e['bounds']);frame['inner_j']=e['j'];assembled.append(frame)
        frame=pd.concat(assembled,ignore_index=True).set_index('row_id').loc[es[0]['outer_train_ids']].reset_index()
        frame['prefix_A']=frame.groupby(['farm','day']).A.transform(lambda a:a.expanding().mean())
        expected=pd.read_csv(L/'nested_snapshot'/f'OOF_DIAG10_{k}_{seed}.csv',float_precision='round_trip')
        assert frame.row_id.tolist()==expected.row_id.tolist();assert np.max(abs(frame.A-expected.A))<1e-12
        dest=O/f'train_{k}_{seed}.csv';frame.to_csv(dest,index=False)
        # External q uses its original outer-trained model outputs, not nested/diagnostic new models.
        with np.load(ROOT/f'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/components/DIAG10_{k}_r3_7.npz',allow_pickle=False) as rr:ti=rr['train_row_id'].astype(str);qi=rr['row_id'].astype(str)
        tt,qq,_=S.season(S.ordered(f,ti),S.ordered(f,qi),vec)
        ext=z[(z.fold==k)&(z.seed==seed)].set_index('row_id').loc[qi].reset_index()
        frame=qq[['row_id','farm','day','hour','sub_ec']+S.M.FULL].copy();frame=frame.loc[:,~frame.columns.duplicated()]
        for n,col in [('et','raw_et'),('lgb','raw_lgb'),('mlp','raw_mlp'),('pfn','old_pfn_raw')]:frame['raw_'+n]=ext[col].to_numpy();frame['smooth_'+n]=S.M.shrink(ext[col].to_numpy(),qq)
        frame['A']=ext.baseline.to_numpy();frame['prefix_A']=frame.groupby(['farm','day']).A.transform(lambda a:a.expanding().mean())
        frame.to_csv(O/f'query_{k}_{seed}.csv',index=False)
        assert not set(frame.row_id)&set(expected.row_id)
        manifest.append(dict(k=k,seed=seed,train_rows=len(expected),query_rows=len(frame),train_sha=sha(dest),query_sha=sha(O/f'query_{k}_{seed}.csv')))
save=dict(status='PASS_PARTIAL_INPUT_RECONSTRUCTION',files=manifest,outer_folds=4,full80=False,fit=0,source_sha=sha(__file__),model_training=False)
(H/'risk_input_preparation_v1.json').write_text(json.dumps(save,indent=2),encoding='utf-8');print('RISK INPUTS VERIFIED',len(manifest),flush=True)
