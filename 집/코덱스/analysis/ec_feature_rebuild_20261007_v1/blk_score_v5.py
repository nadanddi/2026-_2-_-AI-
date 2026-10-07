"""Standalone pinned scorer v3; original statistics plus verified-source rechecks."""
"""Paired BLK diagnosis. Requires verified whole-baseline receipt before target read."""
from pathlib import Path
import csv,json,hashlib,math,random
from collections import defaultdict
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rmse(pred,target,indices):return math.sqrt(math.fsum((pred[i]-target[i])**2 for i in indices)/len(indices)) if indices else None
def quantile(v,p):
    s=sorted(v);x=(len(s)-1)*p;lo=int(x);hi=min(lo+1,len(s)-1)
    return s[lo]+(s[hi]-s[lo])*(x-lo)

def main():
    folder=HERE/'checkpoints/BLK_ASSEMBLED_REFONLY_v1'
    receipt=HERE/'BLK_verified_baseline_receipt_v3.json'
    gate=json.loads(receipt.read_text(encoding='utf-8'))
    assert gate['status']=='VERIFIED_BASELINE_FOR_BLK_DIAGNOSTIC' and gate['whole_pipeline_gate_passed']
    assert gate['code_sha256']==sha(HERE/'verify_BLK_baseline_v3.py')
    assert all(sha(HERE/n)==v for n,v in gate['verifier_lineage_sha256'].items())
    assert all(sha(HERE/n)==v for n,v in gate['verified_evidence_sha256'].items())

    cached_path=HERE/'BLK_cached_PFN_receipt_v2.json'
    assert gate['cached_PFN_receipt_sha256']==sha(cached_path)
    assert gate['cached_PFN_helper_sha256']==sha(HERE/'blk_cached_pfn_receipt_v2.py')
    cached=json.loads(cached_path.read_text(encoding='utf-8'))
    assert cached['code_sha256']==gate['cached_PFN_helper_sha256']
    assert set(cached['context_receipts'])=={'5','6','7','8'}
    assert all(sha(HERE/n)==s for n,s in cached['receipt_files_sha256'].items())
    assert gate['predictions_sha256']==sha(folder/'predictions.json')
    regpath=HERE/'BLK_method_registration_v4.json';reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert gate['method_registration_sha256']==sha(regpath)
    parent=json.loads((HERE/'BLK_method_registration_v3.json').read_text(encoding='utf-8'))
    for name in ['methods','fixed_unsearched_settings','seeds','normal_high_split','position','statistics','screened_variant_count']:
        assert reg[name]==parent[name], ('Frozen method/statistic changed',name)
    assert reg['screened_variant_count']==6 and reg['seeds']==[47,1414,6464]
    assert reg['statistics']['draws']==20000 and reg['statistics']['rng_seed']==2026100702
    assert reg['statistics']['comparison_alpha']==.025/6
    predictions=json.loads((folder/'predictions.json').read_text(encoding='utf-8'))
    assert not predictions['hidden_truth_loaded']
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    ids=predictions['row_ids'];assert len(ids)==len(set(ids))==1440 and set(ids)==set(layout['query_ids'])
    assert sha(ROOT/'공용/대회자료/정형데이터/참가자_배포/train_y.csv')==layout['source_sha256']['train_y.csv']
    spec={'method_registration_sha256':sha(regpath),'baseline_verified_receipt_sha256':sha(receipt),'prediction_sha256':sha(folder/'predictions.json'),
        'code_sha256':sha(__file__),'source_template_sha256':sha(HERE/'blk_score_v1.py'),'parent_scorer_sha256':sha(HERE/'blk_score_v3.py'),'previous_cached_scorer_sha256':sha(HERE/'blk_score_v4.py'),'cached_PFN_receipt_sha256':sha(cached_path),'bootstrap_engine':'Python3.12 random.Random MT19937 randrange','seed':2026100702,'draws':20000,
        'loss':'per-row mean_seed[(candidate-y)^2-(baseline-y)^2]','sampling':'4 blocks with replacement within each of2 farms, row-weighted sum SSE/count',
        'CI':'linear interpolated empirical2.5/97.5 percentiles, individual descriptive95%, not multiplicity adjusted',
        'p':'(1+count(delta>=0))/20001','alpha':.025/6,'adoption_permitted':False,'diagnostic_only':True}
    sp=HERE/'BLK_scorer_spec_v1.json';assert not sp.exists()
    sp.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    # Held-out EC is read only after all contracts and whole-model gate above.
    truth={}
    allowed_score_ids=set(ids)
    with (ROOT/'공용/대회자료/정형데이터/참가자_배포/train_y.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['row_id'] in allowed_score_ids:
                assert row['row_id'] not in truth, 'Duplicate query truth ID'
                value=float(row['sub_ec']); assert math.isfinite(value), 'Nonfinite query truth'
                truth[row['row_id']]=value
    assert set(truth)==set(ids)
    y=[truth[r] for r in ids];positions={};block_index={};farm_blocks=defaultdict(list)
    for bindex,b in enumerate(layout['blocks']):
        farm_blocks[b['farm']].append(bindex)
        for i,d in enumerate(b['query_days']):
            for h in range(24):
                rid=f'{b["farm"]}_{d:03d}_{h:02d}';positions[rid]=['앞','가운데','뒤'][3*i//len(b['query_days'])];block_index[rid]=bindex
    assert set(block_index)==set(ids) and all(len(v)==4 for v in farm_blocks.values())
    daily=defaultdict(list)
    for rid,value in truth.items():daily[rid[:7]].append(value)
    assert len(daily)==60 and all(len(v)==24 for v in daily.values())
    high={d:math.fsum(v)/len(v)>=1 for d,v in daily.items()}
    subsets={'전체':list(range(len(ids))),'일반':[i for i,r in enumerate(ids) if not high[r[:7]]],'고EC':[i for i,r in enumerate(ids) if high[r[:7]]]}
    subsets.update({name:[i for i,r in enumerate(ids) if positions[r]==name] for name in ['앞','가운데','뒤']})
    block_rows=[[i for i,r in enumerate(ids) if block_index[r]==b] for b in range(8)]
    rng=random.Random(2026100702)
    samples=[[rng.choice(farm_blocks[f]) for f in ['F13','F47'] for _ in range(4)] for _ in range(20000)]
    results=[]
    for scope in ['BLK_QUERY_ROLE','BLK_RAW_PASS']:
        for method in ['PAST_ENDPOINT','BOTH_ENDPOINT','CHAIN_PREFIX_GUARD']:
            bs=[predictions['baseline'][f'{scope}_seed{s}'] for s in [47,1414,6464]]
            cs=[predictions['candidate'][f'{scope}_seed{s}'][method] for s in [47,1414,6464]]
            assert all(len(v)==len(ids) and all(math.isfinite(x) for x in v) for v in bs+cs)
            cells=[]
            for j,s in enumerate([47,1414,6464]):
                for segment,index in subsets.items():
                    base=rmse(bs[j],y,index);candidate=rmse(cs[j],y,index)
                    cells.append({'seed':s,'segment':segment,'rows':len(index),'baseline_RMSE':base,'candidate_RMSE':candidate,
                        'delta_RMSE':candidate-base if index else None})
            row_delta=[math.fsum((cs[j][i]-y[i])**2-(bs[j][i]-y[i])**2 for j in range(3))/3 for i in range(len(ids))]
            sse=[math.fsum(row_delta[i] for i in ix) for ix in block_rows];counts=[len(ix) for ix in block_rows]
            boots=[math.fsum(sse[b] for b in sample)/sum(counts[b] for b in sample) for sample in samples]
            p=(1+sum(d>=0 for d in boots))/20001
            overall=[c for c in cells if c['segment']=='전체']
            all_improve=all(c['delta_RMSE']<0 for c in overall)
            results.append({'scope':scope,'method':method,'mean_seed_delta_RMSE':math.fsum(c['delta_RMSE'] for c in overall)/3,
                'all_three_seeds_improve':all_improve,'p_worse':p,'individual95_MSE_delta_CI':[quantile(boots,.025),quantile(boots,.975)],
                'BLK_screen_pass':all_improve and p<.025/6,'adopted':False,'cells':cells,'block_MSE_delta_sums':sse})
    results.sort(key=lambda r:(r['scope'],r['mean_seed_delta_RMSE']))
    out=HERE/'BLK_diagnostic_results_v1.json';assert not out.exists()
    out.write_text(json.dumps({'status':'BLK_DIAGNOSIS_ONLY','rows':len(ids),'blocks':8,'scorer_spec_sha256':sha(sp),'high_rows':len(subsets['고EC']),
        'results':results,'whole_goal_complete':False,'adoption_permitted':False,
        'remaining':['TM/P2LOO/EL1 originalfold effect checks','newseed/layout single confirmation','domain/literature/data feature stages']},ensure_ascii=False,indent=2),encoding='utf-8')
    cp=HERE/'BLK_diagnostic_cells_v1.csv';assert not cp.exists()
    with cp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['scope','method','seed','segment','rows','baseline_RMSE','candidate_RMSE','delta_RMSE']);writer.writeheader()
        for result in results:
            for cell in result['cells']:writer.writerow({'scope':result['scope'],'method':result['method'],**cell})
    print(json.dumps([{k:r[k] for k in ['scope','method','mean_seed_delta_RMSE','p_worse','BLK_screen_pass']} for r in results]),flush=True)

if __name__=='__main__':main()
