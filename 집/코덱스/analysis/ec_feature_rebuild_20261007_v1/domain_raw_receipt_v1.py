"""Strict raw ET metadata/matrices/independent train median checks. No query truth."""
from blk_baseline_data_v1 import *
from domain_features_v2 import build_domain
from run_domain_BLK_raw_v3 import matrix_sha
from checkpoint_v1 import digest,atomic
import json,hashlib,math,warnings,argparse
import sklearn

def verify_raw(require_complete=True):
    folder=HERE/'checkpoints/DOMAIN24_BLK_RAW_v3'
    regpath=HERE/'DOMAIN24_BLK_fit_registration_v3.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status']=='REGISTERED_BEFORE_DOMAIN_FIT'
    assert reg['runner_sha256']==sha(HERE/'run_domain_BLK_raw_v3.py')
    assert all(sha(Path(path))==value for path,value in reg['source_sha256'].items())
    expected_files={f'{c}_seed{s}.json' for c in reg['family_map'] for s in reg['seeds']}
    snapshot={path.name:path for path in folder.glob('D??_seed*.json')}
    assert snapshot and set(snapshot)<=expected_files
    complete_path=folder/'complete.json'
    complete=None
    if require_complete:
        complete=json.loads(complete_path.read_text(encoding='utf-8'))
        assert set(snapshot)==expected_files and len(snapshot)==72
        assert complete['status']=='DOMAIN24_RAW_ET_COMPLETE_NOT_SCORED'
        assert complete['registration_sha256']==sha(regpath)
        assert complete['raw_candidate_fits']==72 and complete['baseline_replays']==3
        assert complete['heldout_truth_loaded'] is False
        expected_all=expected_files|{f'baseline_ET_seed{s}_replay.json' for s in reg['seeds']}
        assert set(complete['files_sha256'])==expected_all
        assert all(sha(folder/name)==value for name,value in complete['files_sha256'].items())
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    ctx=BLKContext(layout)
    _,tr,calendar=prepare_reference(ctx)
    base_reg=json.loads((HERE/'checkpoints/BLK_R3_v1/registration.json').read_text(encoding='utf-8'))
    assert sklearn.__version__==base_reg['environment']['sklearn']
    assert np.__version__==base_reg['environment']['numpy'] and pd.__version__==base_reg['environment']['pandas']
    ids=base_reg['ordered_query_ids']
    assert tr.row_id.tolist()==base_reg['ordered_train_ids']
    q=prepare_query(ctx,ids,calendar)
    features,selected,_=build_domain(dataframe({**ctx.reference_inputs,**ctx._query}))
    assert set(selected)==set(reg['family_map'])
    tb=tr[FULL_R3].copy();tb.index=tr.row_id
    qb=q[FULL_R3].copy();qb.index=q.row_id
    replay_hashes={}
    for seed in reg['seeds']:
        path=folder/f'baseline_ET_seed{seed}_replay.json'
        if not path.exists():
            assert not require_complete
            continue
        r=json.loads(path.read_text(encoding='utf-8'))
        assert r['status']=='PASS' and r['seed']==seed and r['registration_sha256']==sha(regpath)
        assert 0<=r['max_difference']<=1e-6 and r['hidden_truth_loaded'] is False
        assert r['baseline_receipt_sha256']==sha(HERE/f'checkpoints/BLK_R3_v1/ET_seed{seed}.json')
        assert r['train_matrix_sha256']==matrix_sha(tb) and r['query_matrix_sha256']==matrix_sha(qb)
        replay_hashes[path.name]=sha(path)
    matrices={};medians={};empty={};aliases={}
    for candidate in {name.split('_')[0] for name in snapshot}:
        columns=[c for c in selected[candidate] if c not in FULL_R3]
        assert columns==reg['family_map'][candidate]['additional_ET_columns']
        tx=tb.join(features[columns]);qx=qb.join(features[columns])
        matrices[candidate]={'train_matrix_sha256':matrix_sha(tx),'query_matrix_sha256':matrix_sha(qx),'columns':list(tx.columns)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            # Different median implementation from sklearn's masked-array median.
            median=np.nanmedian(tx.to_numpy(float),axis=0)
        medians[candidate]=hashlib.sha256(median.tobytes()).hexdigest()
        empty[candidate]=[c for c in tx if tx[c].isna().all()]
        aliases[candidate]={c:c.removesuffix('__mean1').removesuffix('__sum1') for c in columns if c.endswith(('__mean1','__sum1'))}
    hashes=dict(replay_hashes);maximum=0.
    for name,path in sorted(snapshot.items()):
        record=json.loads(path.read_text(encoding='utf-8'))
        candidate=record['candidate'];seed=record['seed']
        assert name==f'{candidate}_seed{seed}.json' and seed in reg['seeds']
        assert record['member']=='ET' and record['registration_sha256']==sha(regpath)
        assert record['row_ids']==ids and len(record['pred'])==1440
        assert all(math.isfinite(value) for value in record['pred'])
        assert record['pred_sha256']==digest(record['pred'])
        assert record['hidden_truth_loaded'] is False and record['matrices']==matrices[candidate]
        audit=record['audit']
        assert audit['status']=='PASS' and audit['fit_rows']==5520 and audit['query_rows']==1440
        assert audit['imputer_fit_reference_only'] is True
        differences=audit['reverse_scattered_single_max_differences']
        assert len(differences)==3 and all(math.isfinite(v) and 0<=v<=1e-6 for v in differences)
        maximum=max(maximum,*differences)
        assert record['imputer_statistics_sha256']==medians[candidate],('Independent train median mismatch',candidate,seed)
        assert record['all_missing_train_columns']==empty[candidate]
        expected_output=[c for c in matrices[candidate]['columns'] if c not in empty[candidate]]
        assert record['imputer_output_columns']==expected_output
        assert record['imputer_output_width']==len(expected_output)
        assert record['structural_aliases']==aliases[candidate]
        hashes[name]=sha(path)
    assert all(sha(Path(path))==value for path,value in reg['source_sha256'].items())
    return {'status':'VERIFIED_DOMAIN24_RAW_RECEIPTS' if require_complete else 'PARTIAL_RAW_RECEIPTS_CHECKED_NO_SCORE_GATE',
        'full_raw_gate':require_complete,'registration_sha256':sha(regpath),
        'complete_sha256':sha(complete_path) if complete is not None else None,
        'snapshot_candidate_fits':len(snapshot),'baseline_replays_checked':len(replay_hashes),
        'files_sha256':hashes,'batch_audit_max_difference':maximum,
        'independent_median_method':'numpy.nanmedian train-only SHA vs producer sklearn masked-array median SHA',
        'candidate_train_median_sha256':medians,'empty_train_columns':empty,
        'code_sha256':sha(__file__),'heldout_truth_loaded':False,'model_fit':False,
        'limits':['Raw prediction generation is source/receipt checked, not independently refit here',
                  'Feature prefix audits plus rowwise ET pipeline source/batch tests; final postprocess gate separate']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    receipt=verify_raw(not args.partial)
    out=HERE/('DOMAIN24_raw_partial_check_v1.json' if args.partial else 'DOMAIN24_raw_receipt_v1.json')
    assert not out.exists();atomic(out,json.dumps(receipt,ensure_ascii=False,allow_nan=False))
    print(f"{receipt['status']}: {receipt['snapshot_candidate_fits']} candidate receipts, independent training medians match",flush=True)
