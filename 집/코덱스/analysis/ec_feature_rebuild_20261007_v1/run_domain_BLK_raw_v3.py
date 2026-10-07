"""CPU ET family additions, all24 x3seeds; preserve raw receipts, no scoring."""
from blk_baseline_data_v1 import *
from domain_features_v2 import build_domain
from blk_r3_baseline_v1 import model
from checkpoint_v1 import atomic, digest
from checkpoint_v2 import start_ticks
from threadpoolctl import threadpool_limits
import json, os, time, hashlib, platform
import sklearn

def matrix_sha(frame):
    values=frame.to_numpy(dtype='<f8',copy=True)
    values[np.isnan(values)]=np.nan
    return hashlib.sha256(values.tobytes()).hexdigest()

def verify_sources(reg):
    assert all(sha(Path(path))==value for path,value in reg['source_sha256'].items()), 'Frozen domain source changed'

def main():
    regpath=HERE/'DOMAIN24_BLK_fit_registration_v3.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status']=='REGISTERED_BEFORE_DOMAIN_FIT'
    assert reg['seeds']==[47,1414,6464] and reg['candidate_cap']==24
    assert reg['runner_sha256']==sha(__file__)
    verify_sources(reg)
    prep=json.loads((HERE/'DOMAIN24_BLK_preparation_v2.json').read_text(encoding='utf-8'))
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    ctx=BLKContext(layout)
    _,tr,calendar=prepare_reference(ctx)
    query_ids=json.loads((HERE/'checkpoints/BLK_R3_v1/registration.json').read_text(encoding='utf-8'))['ordered_query_ids']
    q=prepare_query(ctx,query_ids,calendar)
    features,selected,_=build_domain(dataframe({**ctx.reference_inputs,**ctx._query}))
    assert set(selected)==set(prep['family_map'])==set(reg['family_map'])
    assert prep['family_map']==reg['family_map']
    assert sha(HERE/'DOMAIN24_BLK_preparation_v2.json')==reg['preparation_sha256']
    assert sorted(tr.row_id)==prep['train_ids'] and sorted(q.row_id)==prep['query_ids']
    assert set(tr.row_id).isdisjoint(q.row_id)
    train_base=tr[FULL_R3].copy();train_base.index=tr.row_id
    query_base=q[FULL_R3].copy();query_base.index=q.row_id
    y=tr.sub_ec.to_numpy(float)
    folder=HERE/'checkpoints/DOMAIN24_BLK_RAW_v3';folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'RUN_WRITER_LOCK.json'
    token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'registration_sha256':sha(regpath)}
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w',encoding='utf-8') as handle:json.dump(token,handle)
    paths=[]
    try:
        for seed in reg['seeds']:
            # Full baseline ET refit on same data, prior raw receipt reproduced first.
            replay_path=folder/f'baseline_ET_seed{seed}_replay.json'
            if not replay_path.exists():
                verify_sources(reg)
                old=json.loads((HERE/f'checkpoints/BLK_R3_v1/ET_seed{seed}.json').read_text(encoding='utf-8'))
                assert old['row_ids']==query_ids
                m=model('ET',seed)
                with threadpool_limits(limits=1):
                    m.fit(train_base,y);m.steps[-1][1].n_jobs=1
                    replay=np.asarray(m.predict(query_base),float)
                difference=float(np.max(np.abs(replay-np.asarray(old['pred']))))
                assert difference<=1e-6
                atomic(replay_path,json.dumps({'status':'PASS','seed':seed,'registration_sha256':sha(regpath),
                    'max_difference':difference,'baseline_receipt_sha256':sha(HERE/f'checkpoints/BLK_R3_v1/ET_seed{seed}.json'),
                    'train_matrix_sha256':matrix_sha(train_base),'query_matrix_sha256':matrix_sha(query_base),
                    'hidden_truth_loaded':False},allow_nan=False))
                print(f'Baseline ET seed{seed} replay PASS max{difference}',flush=True)
            saved_replay=json.loads(replay_path.read_text(encoding='utf-8'))
            assert saved_replay['registration_sha256']==sha(regpath) and saved_replay['status']=='PASS'
            assert saved_replay['seed']==seed and saved_replay['max_difference']<=1e-6
            assert saved_replay['baseline_receipt_sha256']==sha(HERE/f'checkpoints/BLK_R3_v1/ET_seed{seed}.json')
            assert saved_replay['train_matrix_sha256']==matrix_sha(train_base)
            assert saved_replay['query_matrix_sha256']==matrix_sha(query_base)
            assert saved_replay['hidden_truth_loaded'] is False
            paths.append(replay_path)
            for candidate in sorted(selected):
                out=folder/f'{candidate}_seed{seed}.json'
                expected=prep['family_map'][candidate]['additional_ET_columns']
                assert expected==[c for c in selected[candidate] if c not in FULL_R3]
                tx=train_base.join(features[expected]);qx=query_base.join(features[expected])
                signature={'train_matrix_sha256':matrix_sha(tx),'query_matrix_sha256':matrix_sha(qx),
                           'columns':list(tx.columns)}
                if out.exists():
                    saved=json.loads(out.read_text(encoding='utf-8'))
                    assert saved['registration_sha256']==sha(regpath) and saved['row_ids']==query_ids
                    assert saved['matrices']==signature and saved['pred_sha256']==digest(saved['pred'])
                    assert saved['candidate']==candidate and saved['seed']==seed and saved['member']=='ET'
                    assert len(saved['pred'])==len(query_ids) and np.isfinite(saved['pred']).all()
                    assert saved['hidden_truth_loaded'] is False
                    assert saved['audit']['status']=='PASS' and saved['audit']['fit_rows']==len(tx) and saved['audit']['query_rows']==len(qx)
                    assert saved['audit']['imputer_fit_reference_only'] is True
                    assert len(saved['audit']['reverse_scattered_single_max_differences'])==3
                    assert all(0<=v<=1e-6 for v in saved['audit']['reverse_scattered_single_max_differences'])
                    empty=[c for c in tx if tx[c].isna().all()]
                    assert saved['all_missing_train_columns']==empty
                    assert saved['imputer_output_columns']==[c for c in tx if c not in empty]
                    assert saved['imputer_output_width']==len(saved['imputer_output_columns'])
                    paths.append(out);continue
                verify_sources(reg)
                started=time.monotonic()
                print(f'Domain {candidate} seed{seed} ET fitting CPU, additional{len(expected)}',flush=True)
                m=model('ET',seed)
                with threadpool_limits(limits=1):
                    m.fit(tx,y);m.steps[-1][1].n_jobs=1
                    predicted=np.asarray(m.predict(qx),float)
                    reversed_pred=np.asarray(m.predict(qx.iloc[::-1]),float)[::-1]
                    probe_indices=np.linspace(0,len(qx)-1,32,dtype=int)
                    scattered=np.asarray(m.predict(qx.iloc[probe_indices]),float)
                    single=np.array([float(m.predict(qx.iloc[[i]])[0]) for i in probe_indices[:8]])
                differences=[float(np.max(np.abs(predicted-reversed_pred))),
                             float(np.max(np.abs(predicted[probe_indices]-scattered))),
                             float(np.max(np.abs(predicted[probe_indices[:8]]-single)))]
                assert max(differences)<=1e-6 and np.isfinite(predicted).all()
                imputer=m.steps[0][1]
                verify_sources(reg)
                record={'registration_sha256':sha(regpath),'candidate':candidate,'seed':seed,
                        'member':'ET','row_ids':query_ids,'pred':predicted.tolist(),
                        'pred_sha256':digest(predicted.tolist()),'matrices':signature,
                        'imputer_statistics_sha256':hashlib.sha256(imputer.statistics_.tobytes()).hexdigest(),
                        'imputer_output_columns':imputer.get_feature_names_out().tolist(),
                        'imputer_output_width':int(imputer.transform(tx.iloc[:1]).shape[1]),
                        'structural_aliases':{c:c.removesuffix('__mean1').removesuffix('__sum1') for c in expected
                                              if c.endswith(('__mean1','__sum1'))},
                        'all_missing_train_columns':[c for c in tx if tx[c].isna().all()],
                        'audit':{'status':'PASS','reverse_scattered_single_max_differences':differences,
                                 'fit_rows':len(tx),'query_rows':len(qx),'imputer_fit_reference_only':True},
                        'hidden_truth_loaded':False,'duration_seconds':time.monotonic()-started}
                atomic(out,json.dumps(record,ensure_ascii=False,allow_nan=False))
                paths.append(out)
                print(f'{candidate} seed{seed} raw complete {record["duration_seconds"]:.1f}s',flush=True)
        assert len(paths)==75
        complete=folder/'complete.json'
        verify_sources(reg)
        assert not complete.exists()
        atomic(complete,json.dumps({'status':'DOMAIN24_RAW_ET_COMPLETE_NOT_SCORED','registration_sha256':sha(regpath),
            'files_sha256':{p.name:sha(p) for p in paths},'raw_candidate_fits':72,'baseline_replays':3,
            'heldout_truth_loaded':False,'remaining':['mix/shrink/clip/SG2 full candidate causal audit',
                                                     'registered diagnostic scorer and independent checks',
                                                     'original TM/P2LOO/EL1 all24 candidate tests']},allow_nan=False))
    finally:
        if json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()

if __name__=='__main__':main()


