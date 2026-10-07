"""Frozen domain24 BLK diagnostics; whole gate before query truth conversion."""
from pathlib import Path
import csv,json,hashlib,math,random,sys
from collections import defaultdict
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rmse(values,truth,indices):
    return math.sqrt(math.fsum((values[i]-truth[i])**2 for i in indices)/len(indices)) if indices else None
def quantile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p;low=int(position)
    return ordered[low]+(ordered[min(low+1,len(ordered)-1)]-ordered[low])*(position-low)

def main():
    assert sys.version_info[:2]==(3,12)
    gatepath=HERE/'DOMAIN24_verified_gate_v1.json'
    gate=read(gatepath)
    assert gate['status']=='VERIFIED_DOMAIN24_BLK_FOR_DIAGNOSTIC' and gate['whole_pipeline_gate_passed'] is True
    assert gate['adoption_permitted'] is False and gate['heldout_truth_loaded'] is False
    assert gate['code_sha256']==sha(HERE/'verify_domain_BLK_v1.py')
    assert all(Path(path).is_absolute() and sha(path)==value for path,value in gate['transitive_sources_sha256'].items())
    assert len(gate['transitive_sources_sha256'])>=160
    regpath=HERE/'DOMAIN24_BLK_fit_registration_v3.json'
    pipelinepath=HERE/'DOMAIN24_BLK_pipeline_registration_v2.json'
    reg=read(regpath);pipeline=read(pipelinepath)
    assert gate['fit_registration_sha256']==sha(regpath)
    assert gate['pipeline_registration_sha256']==sha(pipelinepath)
    assert pipeline['fit_registration_sha256']==sha(regpath)
    assert pipeline['statistics']==reg['statistics']
    assert all(sha(path)==value for path,value in pipeline['sources_sha256'].items())
    assert all(sha(path)==value for path,value in reg['source_sha256'].items())
    assert reg['seeds']==[47,1414,6464] and set(reg['family_map'])=={f'D{i:02d}' for i in range(1,25)}
    stats=reg['statistics']
    assert stats['variant_count']==48 and stats['cumulative_current_BLK_variant_count']==54
    assert stats['comparison_alpha']==.025/54 and stats['draws']==20000 and stats['rng_seed']==2026100702
    folder=HERE/'checkpoints/DOMAIN24_BLK_ASSEMBLED_v2'
    predpath=folder/'predictions.json';auditpath=folder/'audit.json'
    assert gate['predictions_sha256']==sha(predpath) and gate['assembly_audit_sha256']==sha(auditpath)
    assert gate['raw_receipt_sha256']==sha(HERE/'DOMAIN24_raw_receipt_v1.json')
    assert gate['raw_helper_sha256']==sha(HERE/'domain_raw_receipt_v1.py')
    predictions=read(predpath);layout=read(HERE/'BLK_layout_v2.json')
    assert predictions['hidden_truth_loaded'] is False
    ids=predictions['row_ids']
    assert len(ids)==len(set(ids))==1440 and set(ids)==set(layout['query_ids'])
    for tag,values in predictions['baseline'].items():
        assert len(values)==1440 and all(math.isfinite(v) for v in values)
        assert set(predictions['candidate'][tag])==set(reg['family_map'])
        assert all(len(v)==1440 and all(math.isfinite(x) for x in v) for v in predictions['candidate'][tag].values())
    targetpath=ROOT/'공용/대회자료/정형데이터/참가자_배포/train_y.csv'
    assert sha(targetpath)==layout['source_sha256']['train_y.csv']
    required={str(path.resolve()) for path in [regpath,pipelinepath,predpath,auditpath,HERE/'BLK_layout_v2.json',
              HERE/'domain_features_v2.py',HERE/'domain_sg2_plan_v1.py',HERE/'run_domain_BLK_raw_v3.py',
              HERE/'domain_raw_receipt_v1.py',HERE/'assemble_domain_BLK_v2.py',HERE/'verify_domain_BLK_v1.py',
              HERE/'score_domain_BLK_v1.py',HERE/'DOMAIN24_raw_receipt_v1.json']}
    assert required<=set(gate['transitive_sources_sha256'])
    spec={'code_sha256':sha(__file__),'gate_sha256':sha(gatepath),'predictions_sha256':sha(predpath),
          'registration_sha256':sha(regpath),'statistics':stats,'adoption_permitted':False,
          'scope':'Already exposed BLK diagnostic only; all24 original validators remain',
          'engine':'Python3.12 random.Random MT19937 randrange'}
    specpath=HERE/'DOMAIN24_BLK_scorer_spec_v1.json';assert not specpath.exists()
    specpath.write_text(json.dumps(spec,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # No heldout numeric value conversion before all contracts above pass.
    wanted=set(ids);truth={}
    with targetpath.open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['row_id'] not in wanted:continue
            assert row['row_id'] not in truth
            value=float(row['sub_ec']);assert math.isfinite(value)
            truth[row['row_id']]=value
    assert set(truth)==wanted
    y=[truth[rid] for rid in ids]
    daily=defaultdict(list)
    for rid,value in truth.items():daily[rid[:7]].append(value)
    assert len(daily)==60 and all(len(v)==24 for v in daily.values())
    high={day for day,values in daily.items() if math.fsum(values)/24>=1}
    positions={};blocks={};farm_blocks=defaultdict(list)
    for index,block in enumerate(layout['blocks']):
        farm_blocks[block['farm']].append(index)
        for position,day in enumerate(block['query_days']):
            for h in range(24):
                rid=f'{block["farm"]}_{day:03d}_{h:02d}'
                positions[rid]=['앞','가운데','뒤'][3*position//len(block['query_days'])]
                blocks[rid]=index
    assert set(blocks)==wanted and all(len(v)==4 for v in farm_blocks.values())
    subsets={'전체':list(range(1440)),'일반':[i for i,r in enumerate(ids) if r[:7] not in high],
             '고EC':[i for i,r in enumerate(ids) if r[:7] in high]}
    subsets.update({p:[i for i,r in enumerate(ids) if positions[r]==p] for p in ['앞','가운데','뒤']})
    block_rows=[[i for i,r in enumerate(ids) if blocks[r]==b] for b in range(8)]
    rng=random.Random(2026100702)
    draws=[[rng.choice(farm_blocks[farm]) for farm in ['F13','F47'] for _ in range(4)] for _ in range(20000)]
    results=[]
    for scope in ['BLK_QUERY_ROLE','BLK_RAW_PASS']:
        for candidate in sorted(reg['family_map']):
            baselines=[predictions['baseline'][f'{scope}_seed{s}'] for s in reg['seeds']]
            candidates=[predictions['candidate'][f'{scope}_seed{s}'][candidate] for s in reg['seeds']]
            cells=[]
            for j,seed in enumerate(reg['seeds']):
                for segment,indices in subsets.items():
                    base=rmse(baselines[j],y,indices);predicted=rmse(candidates[j],y,indices)
                    cells.append({'seed':seed,'segment':segment,'rows':len(indices),'baseline_RMSE':base,
                                  'candidate_RMSE':predicted,'delta_RMSE':predicted-base if indices else None})
            deltas=[math.fsum((candidates[j][i]-y[i])**2-(baselines[j][i]-y[i])**2 for j in range(3))/3 for i in range(1440)]
            sse=[math.fsum(deltas[i] for i in row) for row in block_rows]
            counts=[len(row) for row in block_rows]
            boots=[math.fsum(sse[b] for b in sample)/sum(counts[b] for b in sample) for sample in draws]
            p=(1+sum(v>=0 for v in boots))/20001
            overall=[cell for cell in cells if cell['segment']=='전체']
            improve=all(cell['delta_RMSE']<0 for cell in overall)
            results.append({'scope':scope,'candidate':candidate,'family':reg['family_map'][candidate]['family'],
                'mean_seed_delta_RMSE':math.fsum(cell['delta_RMSE'] for cell in overall)/3,
                'all_three_seeds_improve':improve,'p_worse':p,'individual95_MSE_delta_CI':[quantile(boots,.025),quantile(boots,.975)],
                'BLK_screen_pass':improve and p<.025/54,'adopted':False,'cells':cells,'block_MSE_delta_sums':sse})
    results.sort(key=lambda r:(r['scope'],r['mean_seed_delta_RMSE']))
    out=HERE/'DOMAIN24_BLK_diagnostic_results_v1.json';assert not out.exists()
    out.write_text(json.dumps({'status':'DOMAIN24_BLK_DIAGNOSIS_ONLY','scorer_spec_sha256':sha(specpath),
        'rows':1440,'days':60,'high_days':len(high),'high_rows':len(subsets['고EC']),'blocks':8,
        'variant_count':48,'cumulative_current_BLK_variant_count':54,'results':results,
        'adoption_permitted':False,'whole_goal_complete':False,
        'remaining':['All24 TM/P2LOO/EL1 x3seed','independent verification and stage1 critic',
                     'literature24 and data142/remaining finite subset grammar','first-unused seed/layout single final confirmation']},
        ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    cellpath=HERE/'DOMAIN24_BLK_diagnostic_cells_v1.csv';assert not cellpath.exists()
    with cellpath.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['scope','candidate','family','seed','segment','rows','baseline_RMSE','candidate_RMSE','delta_RMSE'])
        writer.writeheader()
        for entry in results:
            for cell in entry['cells']:writer.writerow({k:entry[k] for k in ['scope','candidate','family']}|cell)
    print('Domain24 BLK diagnostic score complete; independent verification required before reporting effects',flush=True)

if __name__=='__main__':main()
