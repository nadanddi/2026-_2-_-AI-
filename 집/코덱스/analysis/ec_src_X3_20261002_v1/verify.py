"""X3 검산: pandas/NumPy 집계와 독립 csv/Decimal/math.fsum."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
sys.dont_write_bytecode=True
import csv,json,math,hashlib
from decimal import Decimal
from collections import defaultdict,Counter
HERE=Path(__file__).resolve().parent

def main():
    lockpath=Path(env.CODEX)/'ec_final_lock'/'locked_days.json'
    locks={(r['farm'],int(r['day'])) for r in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
    raw={};skip=0;dist=Counter();sigdist=Counter()
    with (Path(env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            farm,day,hour=r['row_id'].split('_');key=(farm,int(day))
            if farm not in ['F13','F47']:continue
            if key in locks:skip+=1;continue
            s=r['sub_ec'];v=Decimal(s);n=v.normalize()
            dist[max(0,-v.as_tuple().exponent)]+=1;sigdist[len(n.as_tuple().digits)]+=1
            raw[r['row_id']]=(key,int(hour),s,v)
    oof={}
    with (ROOT/'집'/'코덱스'/'local'/'ec_restart_phase3_20261001_v1'/'oof_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['validator']=='DIAG10':
                assert r['row_id'] not in oof
                assert abs(float(raw[r['row_id']][3])-float(r['sub_ec']))<1e-12
                oof[r['row_id']]=float(r['v2'])
    days=defaultdict(list)
    for rid,(key,hour,s,v) in raw.items():days[key].append((hour,rid,s,v))
    rowout={r['row_id']:r for r in csv.DictReader((HERE/'row_features.csv').open(encoding='utf-8-sig',newline=''))}
    dayout={(r['farm'],int(r['day'])):r for r in csv.DictReader((HERE/'day_features.csv').open(encoding='utf-8-sig',newline=''))}
    totalerr=[];level=[];shape=[];lincount=0;flatcount=0;low=0;digits05=0;grid=0;differences=[]
    for key,vals in sorted(days.items()):
        vals.sort();assert [v[0] for v in vals]==list(range(24))
        numbers=[float(v[3]) for v in vals];errors=[oof[v[1]]-float(v[3]) for v in vals]
        bias=math.fsum(errors)/24;avg=math.fsum(numbers)/24
        lev=bias*bias;sh=math.fsum((v-bias)**2 for v in errors)/24
        level.append(lev);shape.append(sh);totalerr.extend(v*v for v in errors)
        differences.extend([abs(avg-float(dayout[key]['ec_mean'])),abs(lev-float(dayout[key]['level_sqerr'])),abs(sh-float(dayout[key]['shape_mse']))])
        lin=set();flat=set()
        for a in range(19):
            q=[v[3] for v in vals[a:a+6]]
            if all(z==q[0] for z in q):flat.update(range(a,a+6))
            if max(q)-min(q)>=Decimal('.02') and all(abs(q[j+2]-2*q[j+1]+q[j])<=Decimal('.000001') for j in range(4)):lin.update(range(a,a+6))
        lincount+=len(lin);flatcount+=len(flat)
        for j,(_,rid,s,v) in enumerate(vals):
            n=v.normalize();prec=int(len(n.as_tuple().digits)<=12);digit=int(n.as_tuple().digits[-1] in (0,5));gg=int(v%Decimal('.001')==0)
            low+=prec;digits05+=digit;grid+=gg
            r=rowout[rid]
            assert s==r['raw_ec']
            assert prec==int(float(r['low_precision'])) and digit==int(float(r['digit05'])) and gg==int(float(r['grid001']))
            assert (j in lin)==(r['strict_linear']=='True') and (j in flat)==(r['plateau6']=='True')
    got=dict(rows=len(raw),days=len(days),locked_rows_skipped_before_label_access=skip,rmse=math.sqrt(math.fsum(totalerr)/len(totalerr)),
             level_mse=math.fsum(level)/len(level),shape_mse=math.fsum(shape)/len(shape),
             level_sse_fraction=24*math.fsum(level)/math.fsum(totalerr),low_precision_rows=low,digit05_rows=digits05,
             grid001_rows=grid,strict_linear_rows=lincount,plateau_rows=flatcount)
    expected=json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    for k,v in got.items():assert abs(v-expected[k])<1e-12,(k,v,expected[k])
    assert max(differences)<1e-12
    assert {str(float(k)):v for k,v in dist.items()}==expected['raw_decimal_digits']
    assert {str(float(k)):v for k,v in sigdist.items()}==expected['significant_digits']
    report=dict(status='PASS',independent_method='csv+Decimal+math.fsum, independent exact-window scan',numbers=got,
        day_arithmetic_max_abs_difference=max(differences),row_fingerprints_all_equal=True,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'run.py',HERE/'verify.py',HERE/'results.json',HERE/'stats.csv']},
        final_lock_scored=False)
    (HERE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
