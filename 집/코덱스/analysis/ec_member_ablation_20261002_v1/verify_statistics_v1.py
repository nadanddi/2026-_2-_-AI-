"""그룹 집계·부트스트랩·판정·3시드 평균을 독립 재검산한다."""
from pathlib import Path
import csv,json,math
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'집/코덱스/local/ec_member_ablation_20261002_v1'
ARMS=['drop_et','drop_lgb','drop_mlp','drop_pfn']
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    rows=read(OUT/'oof_predictions.csv');evaluation=load(HERE/'evaluation.json')
    boots={(int(r['seed']),r['arm']):r for r in evaluation['bootstrap']}
    independent=[]
    for seed in [7,101,2024]:
        groups=defaultdict(list)
        for r in rows:
            if r['validator']=='DIAG10' and int(r['seed'])==seed:groups[(r['farm'],int(r['block']))].append(r)
        keys=sorted(groups);assert len(keys)==80
        n=np.array([len(groups[k]) for k in keys],float)
        sumsq={arm:np.array([math.fsum((float(r[arm])-float(r['sub_ec']))**2 for r in groups[k]) for k in keys]) for arm in ['v2']+ARMS}
        rng=np.random.default_rng(918);draw=[]
        for farm in ['F13','F47']:
            positions=np.array([i for i,k in enumerate(keys) if k[0]==farm])
            assert len(positions)==40
            draw.append(positions[rng.integers(0,40,size=(20000,40))])
        draw=np.concatenate(draw,axis=1);den=n[draw].sum(axis=1)
        for arm in ARMS:
            dm=(sumsq[arm][draw]-sumsq['v2'][draw]).sum(axis=1)/den
            dr=np.sqrt(sumsq[arm][draw].sum(axis=1)/den)-np.sqrt(sumsq['v2'][draw].sum(axis=1)/den)
            pw=float(np.mean(dm>=0));lo,hi=np.quantile(dr,[.005,.995])
            b=boots[(seed,arm)]
            assert pw==b['p_worse'] and abs(float(lo)-b['ci_rmse_low'])<1e-12 and abs(float(hi)-b['ci_rmse_high'])<1e-12
            independent.append({'seed':seed,'arm':arm,'p_worse':pw,'ci_rmse_low':float(lo),'ci_rmse_high':float(hi)})
    grouped=defaultdict(list)
    for r in rows:grouped[(r['validator'],int(r['validation_fold']),r['row_id'])].append(r)
    average_errors=defaultdict(list)
    for (name,fold,rid),rs in grouped.items():
        assert len(rs)==3
        y=float(rs[0]['sub_ec'])
        for arm in ['v2']+ARMS:average_errors[(name,arm)].append(math.fsum(float(r[arm]) for r in rs)/3-y)
    official={(r['validator'],r['arm']):r['rmse'] for r in evaluation['ensemble']}
    scores=[]
    for (name,arm),e in sorted(average_errors.items()):
        rmse=math.sqrt(math.fsum(z*z for z in e)/len(e));assert abs(rmse-official[(name,arm)])<1e-12
        scores.append({'validator':name,'arm':arm,'rmse_mean3_predictions':rmse,'n':len(e)})
    decisions=[]
    for arm in ARMS:
        cells=[s for s in evaluation['scores'] if s['arm']==arm]
        b=[r for r in independent if r['arm']==arm]
        direction=all(r['delta_rmse']<0 for r in cells)
        confidence=all(r['p_worse']<.005 and r['ci_rmse_high']<0 for r in b)
        old=next(d for d in evaluation['decisions'] if d['arm']==arm)
        assert (direction and confidence)==old['adopted']
        decisions.append({'arm':arm,'cells_improved':sum(s['delta_rmse']<0 for s in cells),'direction_pass':direction,'confidence_pass':confidence,'passed':direction and confidence})
    audits=load(HERE/'reconstruction_audit.json')
    audit_summary={'r3_match_max':max(a['refit_finished_r3_max_difference'] for a in audits),
                   'v2_reconstruction_max':max(a['restored_v2_max_difference'] for a in audits),
                   'inverse_shrink_roundtrip_max':max(a['pfn_inverse_shrink_roundtrip_max_difference'] for a in audits),
                   'phase3_clip_r3_total':sum(a['clip_r3_count'] for a in audits),'phase3_clip_v2_total':sum(a['clip_v2_count'] for a in audits),
                   'pfn_raw_outside_train_range_occurrences':sum(a['pfn_raw_range']['below_train_range']+a['pfn_raw_range']['above_train_range'] for a in audits),
                   'pfn_finished_outside_train_range_occurrences':sum(a['pfn_finished_range']['below_train_range']+a['pfn_finished_range']['above_train_range'] for a in audits),
                   'candidate_clipped_occurrences':{arm:sum(a[arm+'_preclip_range']['below_train_range']+a[arm+'_preclip_range']['above_train_range'] for a in audits) for arm in ARMS}}
    result={'status':'PASS','bootstrap_independent':independent,'ensemble_independent':scores,'decisions_independent':decisions,'audit_summary':audit_summary}
    (HERE/'statistics_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
