"""Final numerical claims independently regrouped from raw CSV with stdlib only."""
from pathlib import Path
import csv,json,math,collections
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
def main():
    verified=json.loads((H/'verification_v1.json').read_text(encoding='utf-8'));assert verified['status']=='PASS_SCALAR_DATA_EXCLUSION_REPLAY';receipt=json.loads((H/'receipt_v1.json').read_text(encoding='utf-8'));groups=collections.defaultdict(list)
    for rec in receipt['files']:
        with (OUT/rec['path']).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        for r in rows:groups[rec['mode'],rec['v'],rec['s'],rec['scope']].append(r)
    with (H/'scores_v1.csv').open(encoding='utf-8',newline='') as f:scores={(r['mode'],r['v'],int(r['s']),r['scope'],r['segment']):r for r in csv.DictReader(f)}
    results=[];pooled=[]
    for key,rows in groups.items():
        days=collections.defaultdict(list)
        for r in rows:days[int(r['k']),r['farm'],r['day']].append(r)
        high={d for d,z in days.items() if math.fsum(float(r['y']) for r in z)/len(z)>=1}
        for segment in ['all','high','ordinary','pass2']:
            z=[r for r in rows if segment=='all' or segment=='high' and (int(r['k']),r['farm'],r['day']) in high or segment=='ordinary' and (int(r['k']),r['farm'],r['day']) not in high or segment=='pass2' and int(r['day'])>=179]
            if not z:continue
            ae=[float(r['A'])-float(r['y']) for r in z];be=[float(r['P'])-float(r['y']) for r in z];sa=math.fsum(e*e for e in ae);sb=math.fsum(e*e for e in be);change=100*(math.sqrt(sb/sa)-1);old=scores[key+(segment,)];assert len(z)==int(old['n']) and abs(change-float(old['change_pct']))<1e-10
            # Distinct variance+bias identity instead of accumulating squared errors again.
            for e,ss in [(ae,sa),(be,sb)]:
                mu=math.fsum(e)/len(e);center=math.fsum((v-mu)**2 for v in e);assert abs(ss-center-len(e)*mu*mu)<1e-9
            if key[1]=='DIAG10':results.append(dict(mode=key[0],seed=key[2],scope=key[3],segment=segment,n=len(z),change_pct=change,mean_residual=math.fsum(-v for v in ae)/len(z),mean_correction=math.fsum(float(r['P'])-float(r['A']) for r in z)/len(z),both_below=sum(float(r['A'])<float(r['y']) and float(r['P'])<float(r['y']) for r in z)))
        # OOF sum per fold and its dispersion, not a confidence interval for independent days.
        if key[3]=='outer':
            folds=collections.defaultdict(list)
            for r in rows:folds[int(r['k'])].append(r)
            changes=[]
            for _,q in folds.items():
                sa=math.fsum((float(r['A'])-float(r['y']))**2 for r in q);sb=math.fsum((float(r['P'])-float(r['y']))**2 for r in q);changes.append(100*(math.sqrt(sb/sa)-1))
            mu=math.fsum(changes)/len(changes);sd=math.sqrt(math.fsum((v-mu)**2 for v in changes)/len(changes));pooled.append(dict(mode=key[0],validator=key[1],seed=key[2],change_pct=float(scores[key+('all',)]['change_pct']),fold_change_mean=mu,fold_change_sd=sd))
    result=dict(status='PASS_RAW_STD_LIB_FRESH_NUMBERS',records=results,validators=pooled,crosschecks=['raw CSV scalar RMSE vs independently aggregated score','per-record-day label grouping high vs groupby','SSE equals centered variance plus squared bias'],gate_decision=verified['next_gate'])
    with (H/'fresh_numbers_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('FRESH_NUMBERS_PASS')
if __name__=='__main__':main()
