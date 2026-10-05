"""Current actual-v2 baseline daily-level/shape error budget, label-used diagnosis."""
from pathlib import Path
import csv,json,math,collections
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local/ec_gate_failure_investigation_20261005_v1'
def main():
    receipt=json.loads((H.parent/'ec_gate_failure_investigation_20261005_v1/verification_v2.json').read_text(encoding='utf-8'));assert receipt['status'].startswith('PASS_DIAGNOSIS_AND_META270');records=[]
    for seed in [7,101,2024]:
        rows=[]
        for p in sorted(OUT.glob(f'outer_actual_LR_DIAG10_*_{seed}.csv')):
            with p.open(encoding='utf-8',newline='') as f:rows.extend(csv.DictReader(f))
        assert len(rows)==8640 and len({r['row_id'] for r in rows})==8640;days=collections.defaultdict(list)
        for r in rows:days[r['farm'],r['day']].append(r)
        level=[];shape=[];direct=[];high_level=[];ordinary_level=[];high_sse=[];ordinary_sse=[]
        for _,q in days.items():
            assert len(q)==24;e=[float(r['A'])-float(r['y']) for r in q];mu=math.fsum(e)/24;lev=24*mu*mu;sh=math.fsum((v-mu)**2 for v in e);ss=math.fsum(v*v for v in e);assert abs(ss-lev-sh)<1e-11
            oracle=math.fsum((float(r['A'])-mu-float(r['y']))**2 for r in q);assert abs(oracle-sh)<1e-11;level.append(lev);shape.append(sh);direct.append(ss)
            if math.fsum(float(r['y']) for r in q)/24>=1:high_level.append(lev);high_sse.append(ss)
            else:ordinary_level.append(lev);ordinary_sse.append(ss)
        total=math.fsum(direct);lev=math.fsum(level);sh=math.fsum(shape);hl=math.fsum(high_level);ol=math.fsum(ordinary_level);budget=8640*.06**2
        assert abs(total-lev-sh)<1e-9 and abs(lev-hl-ol)<1e-9
        records.append(dict(seed=seed,n=8640,days=len(days),high_days=len(high_level),baseline_rmse=math.sqrt(total/8640),sse=total,level_sse=lev,shape_sse=sh,level_share=lev/total,high_level_share=hl/total,ordinary_level_share=ol/total,perfect_all_day_levels_rmse=math.sqrt(sh/8640),perfect_high_day_levels_rmse=math.sqrt((total-hl)/8640),perfect_ordinary_day_levels_rmse=math.sqrt((total-ol)/8640),target_sse=budget,required_sse_reduction=1-budget/total,remaining_shape_reduction_if_levels_perfect=max(0,1-budget/sh)))
    with (H/'target006_bounds_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_SCALAR_DAY_ORACLE_DECOMPOSITION',purpose='y-used error budget only; not attainable prediction, training feature or score forecast',records=records),f,ensure_ascii=False,indent=2)
    print(json.dumps(records,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
