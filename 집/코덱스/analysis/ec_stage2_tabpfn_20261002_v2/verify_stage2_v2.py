"""K2/K3 공개 OOF 및 캐시 독립 검산: CSV와 math.fsum 기반."""
from pathlib import Path
import sys,csv,json,math,hashlib
sys.dont_write_bytecode=True
import run_stage2_tabpfn as e
import numpy as np

def main():
    common=e.load_common();raw,lab,folds,locks,core=common.prepare()
    labels=dict(zip(lab.row_id,lab.sub_ec))
    with (e.OUT/'oof_long.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    result=json.loads((e.HERE/'result.json').read_text(encoding='utf-8'))
    assert len(rows)==83160
    assert len({(r['validator'],r['validation_fold'],r['seed'],r['row_id']) for r in rows})==len(rows)
    assert all(float(r['sub_ec'])==labels[r['row_id']] for r in rows)
    scores=[];boots=[];maxgap=0.
    for seed in e.R3_SEEDS:
        for name in e.VALIDATORS:
            g=[r for r in rows if int(r['seed'])==seed and r['validator']==name]
            assert g
            cs=math.fsum((float(r['sub_ec'])-float(r['E40_8']))**2 for r in g)
            cr=math.sqrt(cs/len(g))
            for reference in ['v2_fresh','v2_fixed']:
                bs=math.fsum((float(r['sub_ec'])-float(r[reference]))**2 for r in g)
                br=math.sqrt(bs/len(g))
                original=next(z for z in result['scores'] if z['validator']==name and z['seed']==seed and z['reference']==reference)
                maxgap=max(maxgap,abs(cr-original['candidate_rmse']),abs(br-original['reference_rmse']))
                assert maxgap<1e-12 and original['improved']==(cr<br)
                scores.append({'validator':name,'seed':seed,'reference':reference,'candidate_rmse':cr,'reference_rmse':br,'improved':cr<br,'n':len(g)})
                if name!='DIAG10':continue
                assert len(g)==8640 and len({r['row_id'] for r in g})==8640
                groups={}
                for r in g:groups.setdefault((r['farm'],int(r['block'])),[]).append(r)
                units=sorted(groups);assert len(units)==80
                n=np.asarray([len(groups[k]) for k in units],float)
                c=np.asarray([math.fsum((float(r['sub_ec'])-float(r['E40_8']))**2 for r in groups[k]) for k in units])
                b=np.asarray([math.fsum((float(r['sub_ec'])-float(r[reference]))**2 for r in groups[k]) for k in units])
                rng=np.random.default_rng(918);indices=[]
                for farm in ['F13','F47']:
                    choices=np.asarray([i for i,k in enumerate(units) if k[0]==farm]);indices.append(choices[rng.integers(0,len(choices),(20000,len(choices)))])
                ix=np.concatenate(indices,axis=1);dm=(c[ix]-b[ix]).sum(1)/n[ix].sum(1)
                p=float(np.mean(dm>=0));lo,hi=np.quantile(dm,[.0125,.9875])
                orig=next(z for z in result['bootstrap'] if z['seed']==seed and z['reference']==reference)
                assert p==orig['p_worse'] and abs(float(lo)-orig['ci_mse'][0])<1e-12 and abs(float(hi)-orig['ci_mse'][1])<1e-12
                boots.append({'seed':seed,'reference':reference,'p_worse':p,'ci_low':float(lo),'ci_high':float(hi),'passed':p<.0125 and hi<0})
    meta=json.loads((e.OUT/'collection_manifest.json').read_text(encoding='utf-8'))
    base=meta['base_provenance'];r3_count=pfn_count=0
    for fold in folds:
        tr,va=common.split_fold(raw,lab,fold,locks);prov=e.fold_provenance(tr,va,core,base)
        r3,pfn=e.verify_fold(tr,va,core,fold,prov);r3_count+=len(r3);pfn_count+=len(pfn)
    assert r3_count==66 and pfn_count==176
    direction=all(r['improved'] for r in scores);confidence=all(r['passed'] for r in boots)
    assert result['public_selection_gate_pass']==(direction and confidence)
    report={'status':'PASS','occurrences':len(rows),'r3_caches_verified':r3_count,'pfn_caches_verified':pfn_count,
            'max_independent_rmse_gap':maxgap,'scores':scores,'bootstrap':boots,
            'direction_pass':direction,'confidence_pass':confidence,'public_gate_pass':direction and confidence,
            'candidate_adopted':False,'final_lock_scored':False,'K3_data_only':True,
            'oof_sha256':e.sha(e.OUT/'oof_long.csv'),
            'limitation':'PFN 문맥 공유·공개 검증 반복 사용. 이 확인은 동일 고정 안의 새 시드 재검사이며 독립 데이터 수집이 아니다.'}
    e.atomic_json(e.HERE/'independent_verification_v2.json',report)
    print(json.dumps({'status':'PASS','improved_cells':sum(r['improved'] for r in scores),'total_cells':30,'bootstrap':boots},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
