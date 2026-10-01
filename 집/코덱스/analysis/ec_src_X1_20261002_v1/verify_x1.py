from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,math,hashlib
from collections import defaultdict,Counter

HERE=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def avg(xs):return math.fsum(xs)/len(xs)
def main():
    summary=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    locks={(r['farm'],int(r['day'])) for r in json.loads((Path(env.CODEX)/'ec_final_lock/locked_days.json').read_text(encoding='utf-8'))['selected']}
    days=defaultdict(dict);truth={};excluded=0
    for r in read(Path(env.DATA)/'train_y.csv'):
        f,d,h=r['row_id'].split('_');k=(f,int(d));h=int(h)
        if f not in ['F13','F47']:continue
        if k in locks:excluded+=1;continue
        days[k][h]=float(r['sub_ec']);truth[r['row_id']]=float(r['sub_ec'])
    assert excluded==960 and len(days)==summary['days']
    errors=defaultdict(list);rowerrors=[]
    for r in read(ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1/oof_predictions.csv'):
        if r['validator']!='DIAG10' or r['row_id'] not in truth:continue
        f,d,_=r['row_id'].split('_')
        err=float(r['v2'])-truth[r['row_id']]
        errors[(f,int(d))].append(err);rowerrors.append(err)
    level=math.sqrt(avg([avg(v)**2 for v in errors.values()]))
    row=math.sqrt(avg([e*e for e in rowerrors]))
    assert abs(level-summary['v2_day_level_rmse'])<1e-12
    assert abs(row-summary['v2_diag10_row_rmse'])<1e-12
    daylist=read(HERE/'days.csv')
    for r in daylist:
        k=(r['farm'],int(r['day']));vals=list(days[k].values())
        assert k not in locks and set(days[k])==set(range(24))
        assert abs(avg(vals)-float(r['mean_ec']))<1e-12
        assert abs((max(vals)-min(vals))-float(r['range_ec']))<1e-12
    pairs=read(HERE/'all_primary_pairs.csv')
    for p in pairs:
        ki=(p['farm_i'],int(p['day_i']));kj=(p['farm_j'],int(p['day_j']))
        shift=int(p['shift']);a=float(p['a']);b=float(p['b'])
        hours=[h for h in range(24) if 0<=h+shift<24]
        x=[days[kj][h+shift] for h in hours];y=[days[ki][h] for h in hours]
        mx=avg(x);my=avg(y)
        if p['family']=='identity':af=1.;bf=0.
        elif p['family']=='offset':af=1.;bf=my-mx
        elif p['family']=='scale':af=math.fsum(u*v for u,v in zip(x,y))/math.fsum(u*u for u in x);bf=0.
        else:
            af=math.fsum((u-mx)*(v-my) for u,v in zip(x,y))/math.fsum((u-mx)**2 for u in x)
            bf=my-af*mx
        assert abs(a-af)<1e-8 and abs(b-bf)<1e-8
        e=[v-(af*u+bf) for u,v in zip(x,y)]
        rmse=math.sqrt(avg([z*z for z in e]));maxerr=max(abs(z) for z in e)
        diff=abs(avg(list(days[ki].values()))-avg(list(days[kj].values())))
        assert abs(rmse-float(p['rmse']))<1e-9 and abs(maxerr-float(p['maxabs']))<1e-9
        assert rmse<=.00050000001 and maxerr<=.00100100001 and diff<=level/4+1e-12
        assert ki[0]==kj[0] and abs(ki[1]-kj[1])>=7
    selected=read(HERE/'selected_pairs.csv')
    observed=Counter((p['stratum'],p['family']) for p in selected)
    for si,s in enumerate(summary['strata']):
        for fi,family in enumerate(summary['families']):
            assert observed[(s,family)]==summary['observed_independent_pairs'][si][fi]
            chosen=[p for p in selected if p['stratum']==s and p['family']==family]
            all_days=[int(p[col]) for p in chosen for col in ['day_i','day_j']]
            assert all(abs(a-b)>=7 for i,a in enumerate(all_days) for b in all_days[i+1:])
    null=read(HERE/'null_counts.csv');assert len(null)==999
    verified_p=[]
    for fi,f in enumerate(summary['families']):
        total=sum(observed[(s,f)] for s in summary['strata'])
        nulltotal=[sum(int(r[f'{s}/{f}']) for s in summary['strata']) for r in null]
        p=(1+sum(z>=total for z in nulltotal))/1000
        assert p==summary['p_per_family'][fi]
        assert (p<.0125 and all(observed[(s,f)]>=1 for s in summary['strata']))==summary['discovery_per_family'][fi]
        verified_p.append(p)
    hashes={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in summary['hashes'].items()}
    assert all(hashes.values())
    result={'status':'PASS','independent_method':'csv + math.fsum; least-squares independently re-fitted',
            'days':len(days),'rows':len(truth),'lock_rows_skipped_before_float':excluded,'baseline_level_rmse':level,
            'baseline_row_rmse':row,'primary_pairs_checked':len(pairs),'selected_pairs_checked':len(selected),
            'p_verified':verified_p,'hash_checks':hashes,'final_lock_scored':False,'model_training':False}
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
