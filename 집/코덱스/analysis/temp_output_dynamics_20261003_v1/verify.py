from pathlib import Path
import csv,json,math,collections,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
OUT=ROOT/'집/코덱스/local/temp_output_dynamics_20261003_v1'
def main():
    data=json.loads((HERE/'verification.json').read_text(encoding='utf-8'));groups=collections.defaultdict(list)
    with (OUT/'oof.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):groups[(r['validator'],int(r['seed']),r['context'])].append(r)
    rng=np.random.default_rng(20261003);checks=[]
    for key,rows in sorted(groups.items()):
        ea=[float(r['base'])-float(r['sub_temp']) for r in rows];eb=[float(r['candidate'])-float(r['sub_temp']) for r in rows]
        a=math.sqrt(math.fsum(e*e for e in ea)/len(ea));b=math.sqrt(math.fsum(e*e for e in eb)/len(eb));stored=next(s for s in data['summary'] if (s['validator'],s['seed'],s['context'])==key)
        assert abs(a-stored['baseline_rmse'])<1e-12 and abs(b-stored['candidate_rmse'])<1e-12
        check=dict(validator=key[0],seed=key[1],context=key[2],baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1))
        if key[0]=='DIAG10':
            blocks=collections.defaultdict(list)
            for r,x,y in zip(rows,ea,eb):blocks[r['farm']+'_'+str(int(r['day'])//5)].append(y*y-x*x)
            sums=[math.fsum(blocks[k]) for k in sorted(blocks)];counts=[len(blocks[k]) for k in sorted(blocks)];ix=rng.integers(0,len(sums),size=(20000,len(sums)))
            means=np.array([math.fsum(sums[int(i)] for i in indices)/sum(counts[int(i)] for i in indices) for indices in ix]);ci=np.quantile(means,[.005,.995]).tolist();pw=float(np.mean(means>=0))
            assert np.allclose(ci,stored['ci99'],atol=1e-12,rtol=0) and pw==stored['p_worse'];check.update(p_worse=pw,ci99=ci)
        checks.append(check)
    out=dict(status='PASS',cells=checks,verdict=data['verdict'],formula_and_causality_cases_from_preregistered_run=data['causality_cases'])
    (HERE/'independent_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':'PASS','verdict':data['verdict']}))
if __name__=='__main__':main()
