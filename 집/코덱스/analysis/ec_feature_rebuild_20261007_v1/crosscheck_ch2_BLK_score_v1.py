"""Independent Decimal60 factorized-loss, RMSE and bootstrap verification of all six CH2 comparisons."""
from pathlib import Path
from decimal import Decimal,localcontext
import json,csv,hashlib,random,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def D(value):return Decimal(str(value))
def close(actual,expected,tolerance=Decimal('1e-12')):
    assert abs(D(actual)-expected)<tolerance,(actual,str(expected))
def percentile(ordered,p):
    pos=D(len(ordered)-1)*p;lower=int(pos);upper=min(lower+1,len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(pos-lower)

def main():
    gate=read('CH2_BLK_verified_gate_v2.json')
    assert gate['status']=='CH2_FULL_INFERENCE_GATE_VERIFIED_FOR_DIAGNOSTIC_ONLY' and not gate['heldout_truth_loaded']
    assert all(sha(p)==v for p,v in gate['source_sha256'].items())
    resultpath=HERE/'CH2_BLK_diagnostic_results_v1.json';result=read(resultpath.name)
    assert result['status']=='CH2_BLK_DIAGNOSTIC_ONLY' and not result['adoption_permitted']
    assert result['gate_sha256']==sha(HERE/'CH2_BLK_verified_gate_v2.json')
    assert result['scorer_sha256']==sha(HERE/'score_ch2_BLK_v1.py')
    assert result['score_registration_sha256']==sha(HERE/'CH2_BLK_score_registration_v1.json')
    predpath=HERE/'checkpoints/CH2_BLK_v1/predictions.json'
    assert result['predictions_sha256']==gate['predictions_sha256']==sha(predpath)
    pred=json.loads(predpath.read_text(encoding='utf-8'));layout=read('BLK_layout_v2.json')
    ids=pred['row_ids'];wanted=set(ids);labels={}
    with (ROOT/'공용/대회자료/정형데이터/참가자_배포/train_y.csv').open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['row_id'] not in wanted:continue
            assert row['row_id'] not in labels;labels[row['row_id']]=D(row['sub_ec'])
    assert set(labels)==wanted
    blockrid={};positions={};farms={}
    for index,b in enumerate(layout['blocks']):
        farms.setdefault(b['farm'],[]).append(index)
        for di,day in enumerate(b['query_days']):
            for hour in range(24):
                rid=f'{b["farm"]}_{day:03d}_{hour:02d}';blockrid[rid]=index;positions[rid]=['앞','가운데','뒤'][3*di//len(b['query_days'])]
    assert sorted(farms)==['F13','F47'] and all(len(v)==4 for v in farms.values())
    daily={}
    for rid,value in labels.items():daily.setdefault(rid[:7],[]).append(value)
    high={day:sum(values,Decimal(0))/D(len(values))>=1 for day,values in daily.items()}
    def indexes(segment):
        if segment=='전체':return list(range(1440))
        if segment in ['일반','고EC']:return [i for i,r in enumerate(ids) if high[r[:7]]==(segment=='고EC')]
        if segment in ['앞','가운데','뒤']:return [i for i,r in enumerate(ids) if positions[r]==segment]
        if segment in ['F13','F47']:return [i for i,r in enumerate(ids) if r.startswith(segment+'_')]
        assert segment.startswith('hour');return [i for i,r in enumerate(ids) if int(r[-2:])==int(segment[4:])]
    rng=random.Random(2026100702)
    samples=[[farms[f][rng.randrange(4)] for f in ['F13','F47'] for _ in range(4)] for _ in range(20000)]
    reports=[];rmse_checks=0
    with localcontext() as ctx:
        ctx.prec=60
        y=[labels[r] for r in ids]
        for item in result['results']:
            scope=item['scope'];method=item['method'];seedarrays={}
            for seed in [47,1414,6464]:
                name=f'{scope}_seed{seed}'
                seedarrays[seed]=([D(v) for v in pred['baseline'][name]],[D(v) for v in pred['candidate'][name][method]])
            assert len(item['cells'])==96
            for cell in item['cells']:
                ix=indexes(cell['segment']);assert len(ix)==cell['rows']
                base,candidate=seedarrays[cell['seed']]
                if not ix:
                    assert cell['baseline_RMSE'] is None and cell['candidate_RMSE'] is None;continue
                br=(sum(((base[i]-y[i])**2 for i in ix),Decimal(0))/D(len(ix))).sqrt()
                cr=(sum(((candidate[i]-y[i])**2 for i in ix),Decimal(0))/D(len(ix))).sqrt()
                close(cell['baseline_RMSE'],br);close(cell['candidate_RMSE'],cr);close(cell['delta_RMSE'],cr-br);rmse_checks+=2
            # Algebraically factor the SSE difference independently of the scorer's squared-loss subtraction.
            block_sums=[Decimal(0)]*8;counts=[0]*8
            for i,rid in enumerate(ids):
                delta=sum(((cs[i]-bs[i])*(cs[i]+bs[i]-2*y[i]) for bs,cs in seedarrays.values()),Decimal(0))/3
                bi=blockrid[rid];block_sums[bi]+=delta;counts[bi]+=1
            for old,new in zip(item['block_MSE_delta_sums'],block_sums):close(old,new)
            boots=[]
            for sample in samples:
                total=sum((block_sums[b] for b in sample),Decimal(0));n=sum(counts[b] for b in sample)
                boots.append(total/D(n))
            worse=sum(value>=0 for value in boots);p=D(worse+1)/20001;close(item['p_worse'],p)
            boots.sort()
            for actual,prob in zip(item['individual95_MSE_delta_CI'],[D('.025'),D('.975')]):close(actual,percentile(boots,prob))
            for actual,prob in zip(item['multiplicity60_MSE_delta_CI'],[D('.025')/60,1-D('.025')/60]):close(actual,percentile(boots,prob))
            directions=all(c['delta_RMSE']<0 for c in item['cells'] if c['segment']=='전체')
            assert item['all_three_seeds_improve']==directions
            assert item['BLK_screen_pass']==(directions and p<D('.025')/60)
            reports.append({'scope':scope,'method':method,'p_worse_exact_fraction':f'{worse+1}/20001','RMSE_checks':192,'block_and_CI_checks_pass':True})
    assert rmse_checks==1152 and len(reports)==6
    out=HERE/'CH2_BLK_score_independent_crosscheck_v1.json';assert not out.exists()
    report={'status':'DECIMAL60_ALL_CH2_RMSE_FACTORIZED_LOSS_BOOTSTRAP_PASS',
            'result_sha256':sha(resultpath),'predictions_sha256':sha(predpath),'checker_sha256':sha(__file__),
            'gate_sha256':sha(HERE/'CH2_BLK_verified_gate_v2.json'),'RMSE_checks':rmse_checks,
            'variant_count':6,'reports':reports,'adoption_permitted':False,'whole_goal_complete':False}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Independent Decimal60 CH2 all1152RMSE/48blockSSE/6bootstrap p and CI PASS',flush=True)

if __name__=='__main__':main()
