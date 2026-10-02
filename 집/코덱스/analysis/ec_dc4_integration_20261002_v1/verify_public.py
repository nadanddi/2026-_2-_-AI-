"""독립 CSV/표준 라이브러리 산술 및 부트스트랩 검산. 학습 없이 공개 OOF만."""
import sys, csv, json, math
sys.dont_write_bytecode=True
import run_dc4 as d
import numpy as np

def verify(tag,arm):
    path=d.OUT/f'{tag}_oof.csv'
    result=json.loads((d.HERE/f'{tag}_result.json').read_text(encoding='utf-8'))
    with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==83160
    keys=[(r['validator'],r['validation_fold'],r['seed'],r['row_id']) for r in rows]
    assert len(keys)==len(set(keys))
    scores=[];boots=[];maxgap=0.
    for seed in d.SEEDS:
        for name in d.common.VALIDATORS:
            g=[r for r in rows if r['validator']==name and int(r['seed'])==seed]
            assert len(g)>0
            n=len(g);s0=math.fsum((float(r['sub_ec'])-float(r['v2']))**2 for r in g)
            s1=math.fsum((float(r['sub_ec'])-float(r[arm]))**2 for r in g)
            base=math.sqrt(s0/n);cand=math.sqrt(s1/n)
            original=next(z for z in result['scores'] if z['validator']==name and z['seed']==seed and z['arm']==arm)
            gap=abs(cand-original['rmse']);maxgap=max(maxgap,gap);assert gap<1e-12
            assert abs((cand-base)-original['delta_rmse'])<1e-12
            scores.append({'validator':name,'seed':seed,'reference_rmse':base,'candidate_rmse':cand,'n':n,'improved':cand<base})
            if name!='DIAG10':continue
            assert n==8640 and len({r['row_id'] for r in g})==n
            groups={}
            for r in g:groups.setdefault((r['farm'],int(r['block'])),[]).append(r)
            units=sorted(groups);assert len(units)==80
            counts=np.asarray([len(groups[k]) for k in units],float)
            e0=np.asarray([math.fsum((float(r['sub_ec'])-float(r['v2']))**2 for r in groups[k]) for k in units])
            e1=np.asarray([math.fsum((float(r['sub_ec'])-float(r[arm]))**2 for r in groups[k]) for k in units])
            rng=np.random.default_rng(918);draw=[]
            for farm in ['F13','F47']:
                ix=np.asarray([i for i,k in enumerate(units) if k[0]==farm]);draw.append(ix[rng.integers(0,len(ix),(20000,len(ix)))])
            idx=np.concatenate(draw,axis=1)
            dm=(e1[idx]-e0[idx]).sum(1)/counts[idx].sum(1)
            pw=float(np.mean(dm>=0));lo,hi=np.quantile(dm,[.0125,.9875])
            orig=next(z for z in result['bootstrap'] if z['seed']==seed and z['arm']==arm)
            assert pw==orig['p_worse'] and abs(float(lo)-orig['ci_mse_low'])<1e-12 and abs(float(hi)-orig['ci_mse_high'])<1e-12
            boots.append({'seed':seed,'p_worse':pw,'ci_low':float(lo),'ci_high':float(hi)})
    direction=all(r['improved'] for r in scores)
    confident=all(r['p_worse']<.0125 and r['ci_high']<0 for r in boots)
    decision=result['decisions'][0]
    assert decision['direction_pass']==direction and decision['diag_bootstrap_pass']==confident
    if arm=='season_v2':assert decision['adopted']==(direction and confident)
    assert all(r['ET_max_gap']<=1e-8 for r in result['independent_reproduction'])
    evidence={'status':'PASS','csv_sha256':d.engine.sha(path),'occurrences':len(rows),'unique_cells':len(set(keys)),
              'independent_scores':scores,'independent_bootstrap':boots,'max_RMSE_crosscheck_gap':maxgap,
              'direction_pass':direction,'confidence_pass':confident,'lock_labels_opened':False,
              'limitation':'공개 검증 반복 사용 및 PFN 문맥 공유. 정식 후보는 별도 잠금 확인까지 모두 통과해야 함.'}
    d.engine.atomic_json(d.HERE/f'{tag}_independent_verification.json',evidence)
    print(json.dumps({'tag':tag,'status':'PASS','improved_cells':sum(r['improved'] for r in scores),'bootstrap':boots},ensure_ascii=False),flush=True)

if __name__=='__main__':
    verify('et_replication','season_et');verify('v2_integration','season_v2')
