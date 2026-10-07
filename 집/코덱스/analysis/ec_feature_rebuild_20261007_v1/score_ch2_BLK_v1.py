"""Six fixed CH2 diagnostic comparisons; open target values only after the full gate."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,math,random,sys
import run_ch2_BLK_v3 as runner
from ch2_prefix_sources_v3 import METHODS

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def quantile(values,p):
    x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)
def rmse(pred,y,ix):return math.sqrt(math.fsum((pred[i]-y[i])**2 for i in ix)/len(ix)) if ix else None

def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    # First operation requiring a stage artifact: no spec/target parse can precede this gate.
    gate=read('CH2_BLK_verified_gate_v2.json')
    assert gate['status']=='CH2_FULL_INFERENCE_GATE_VERIFIED_FOR_DIAGNOSTIC_ONLY'
    assert gate['diagnostic_scoring_permitted'] and not gate['heldout_truth_loaded']
    assert gate['independent_scalar_checks']==25920 and gate['all_query_prefix_replay_checked']==1440
    assert gate['boundary_checks']==360 and gate['verifier_sha256']==sha(HERE/'verify_ch2_BLK_v2.py')
    assert all(sha(p)==v for p,v in gate['source_sha256'].items())
    spec=read('CH2_BLK_score_registration_v1.json')
    assert spec['scorer_sha256']==sha(__file__) and spec['verifier_sha256']==gate['verifier_sha256']
    reg=read('CH2_BLK_registration_v3.json')
    assert spec['registration_sha256']==gate['registration_sha256']==sha(HERE/'CH2_BLK_registration_v3.json')
    assert reg['methods']==list(METHODS) and reg['scopes']==list(runner.SCOPES) and reg['seeds']==list(runner.SEEDS)
    assert spec['alpha']==reg['statistics']['comparison_alpha']==.025/60
    assert spec['draws']==20000 and spec['rng_seed']==2026100702 and spec['cumulative_comparisons']==60
    assert all(sha(p)==v for p,v in reg['source_sha256'].items())
    path=HERE/'checkpoints/CH2_BLK_v1/predictions.json'
    assert sha(path)==gate['predictions_sha256']
    pred=json.loads(path.read_text(encoding='utf-8'));assert not pred['hidden_truth_loaded']
    baseline,_=runner.baseline_preflight();assert baseline['baseline']==pred['baseline']
    ids=pred['row_ids'];layout=read('BLK_layout_v2.json')
    assert ids==reg['ordered_query_ids'] and set(ids)==set(layout['query_ids']) and len(set(ids))==1440
    truthpath=ROOT/'공용/대회자료/정형데이터/참가자_배포/train_y.csv'
    assert sha(truthpath)==layout['source_sha256']['train_y.csv']
    truth={};wanted=set(ids)
    with truthpath.open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            rid=row['row_id']
            if rid not in wanted:continue
            assert rid not in truth
            truth[rid]=float(row['sub_ec']);assert math.isfinite(truth[rid])
    assert set(truth)==wanted;y=[truth[r] for r in ids]
    block_by_id={};position={};farm_blocks=defaultdict(list)
    for bindex,block in enumerate(layout['blocks']):
        farm_blocks[block['farm']].append(bindex)
        for i,day in enumerate(block['query_days']):
            for hour in range(24):
                rid=f'{block["farm"]}_{day:03d}_{hour:02d}'
                block_by_id[rid]=bindex;position[rid]=['앞','가운데','뒤'][3*i//len(block['query_days'])]
    assert set(block_by_id)==wanted and all(len(v)==4 for v in farm_blocks.values())
    daily=defaultdict(list)
    for rid,value in truth.items():daily[rid[:7]].append(value)
    assert len(daily)==60 and all(len(v)==24 for v in daily.values())
    high={d:math.fsum(values)/24>=1 for d,values in daily.items()}
    segments={'전체':list(range(1440)), '일반':[i for i,r in enumerate(ids) if not high[r[:7]]],
              '고EC':[i for i,r in enumerate(ids) if high[r[:7]]]}
    segments.update({name:[i for i,r in enumerate(ids) if position[r]==name] for name in ['앞','가운데','뒤']})
    segments.update({farm:[i for i,r in enumerate(ids) if r.startswith(farm+'_')] for farm in ['F13','F47']})
    segments.update({f'hour{hour:02d}':[i for i,r in enumerate(ids) if int(r[-2:])==hour] for hour in range(24)})
    assert len(segments)==32
    block_rows=[[i for i,r in enumerate(ids) if block_by_id[r]==b] for b in range(8)]
    rng=random.Random(spec['rng_seed'])
    samples=[[rng.choice(farm_blocks[f]) for f in ['F13','F47'] for _ in range(4)] for _ in range(spec['draws'])]
    results=[]
    for scope in runner.SCOPES:
        for method in METHODS:
            bs=[pred['baseline'][f'{scope}_seed{s}'] for s in runner.SEEDS]
            cs=[pred['candidate'][f'{scope}_seed{s}'][method] for s in runner.SEEDS]
            cells=[]
            for j,seed in enumerate(runner.SEEDS):
                for segment,ix in segments.items():
                    br=rmse(bs[j],y,ix);cr=rmse(cs[j],y,ix)
                    cells.append({'seed':seed,'segment':segment,'rows':len(ix),'baseline_RMSE':br,'candidate_RMSE':cr,
                                  'delta_RMSE':cr-br if ix else None})
            deltas=[math.fsum((cs[j][i]-y[i])**2-(bs[j][i]-y[i])**2 for j in range(3))/3 for i in range(1440)]
            sse=[math.fsum(deltas[i] for i in ix) for ix in block_rows];counts=[len(ix) for ix in block_rows]
            boots=sorted(math.fsum(sse[b] for b in sample)/sum(counts[b] for b in sample) for sample in samples)
            p=(1+sum(v>=0 for v in boots))/20001
            overall=[v for v in cells if v['segment']=='전체'];directions=all(v['delta_RMSE']<0 for v in overall)
            results.append({'scope':scope,'method':method,'mean_seed_delta_RMSE':math.fsum(v['delta_RMSE'] for v in overall)/3,
                            'all_three_seeds_improve':directions,'p_worse':p,'BLK_screen_pass':directions and p<spec['alpha'],
                            'individual95_MSE_delta_CI':[quantile(boots,.025),quantile(boots,.975)],
                            'multiplicity60_MSE_delta_CI':[quantile(boots,.025/60),quantile(boots,1-.025/60)],
                            'block_MSE_delta_sums':sse,'cells':cells,'adopted':False})
    results.sort(key=lambda r:(r['scope'],r['mean_seed_delta_RMSE']))
    output={'status':'CH2_BLK_DIAGNOSTIC_ONLY','scorer_sha256':sha(__file__),'gate_sha256':sha(HERE/'CH2_BLK_verified_gate_v2.json'),
            'score_registration_sha256':sha(HERE/'CH2_BLK_score_registration_v1.json'),
            'predictions_sha256':sha(path),'rows':1440,'blocks':8,'high_rows':len(segments['고EC']),
            'cumulative_comparisons':60,'alpha':.025/60,'supported_rows':gate['supported_rows'],'results':results,
            'adoption_permitted':False,'whole_goal_complete':False,
            'limits':['Fixed exposed exploration layout/pass1 only/high cohort small',
                      'Original TM/P2LOO/EL1 and first-unused seed/layout confirmation still mandatory',
                      'Physical source identity not established; cached CPU baseline only']}
    out=HERE/'CH2_BLK_diagnostic_results_v1.json';assert not out.exists()
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    cellpath=HERE/'CH2_BLK_diagnostic_cells_v1.csv';assert not cellpath.exists()
    with cellpath.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['scope','method','seed','segment','rows','baseline_RMSE','candidate_RMSE','delta_RMSE']);writer.writeheader()
        for result in results:
            for cell in result['cells']:writer.writerow({'scope':result['scope'],'method':result['method'],**cell})
    print('CH2 six diagnostics completed; no adoption; independent arithmetic/bootstrap review required',flush=True)

if __name__=='__main__':main()
