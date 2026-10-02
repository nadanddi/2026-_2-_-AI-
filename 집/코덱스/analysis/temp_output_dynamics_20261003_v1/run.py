from pathlib import Path
import csv,json,math,collections,hashlib
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
OUT=ROOT/'집/코덱스/local/temp_output_dynamics_20261003_v1';OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'집/코덱스/local/temp_season_20261003_v1/oof.csv'
def transform(rows):
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['farm'],int(rows[i]['day']),int(rows[i]['hour'])))
    result=np.array([float(r['base']) for r in rows]);previous=None
    for i in order:
        r=rows[i];t=float(r['in_temp']) if r['in_temp'] else math.nan;g=max(0.,min(1.,(t-8)/2)) if math.isfinite(t) else 0.
        if previous is not None:
            p=rows[previous];pt=float(p['in_temp']) if p['in_temp'] else math.nan
            if (r['farm'],r['day'])==(p['farm'],p['day']) and int(r['hour'])==int(p['hour'])+1 and math.isfinite(pt) and math.isfinite(t):
                assert r['fold']==p['fold'];result[i]-=.2*g*((float(r['base'])-float(p['base']))-.5*(t-pt))
        previous=i
    return result
def main():
    groups=collections.defaultdict(list)
    with SRC.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):groups[(row['validator'],int(row['seed']),row['context'])].append(row)
    summary=[];record=[];rng=np.random.default_rng(20261003)
    with (Path(env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:labels={r['row_id']:float(r['sub_temp']) for r in csv.DictReader(f)}
    for key,rows in sorted(groups.items()):
        assert len({r['row_id'] for r in rows})==len(rows);assert all(labels[r['row_id']]==float(r['sub_temp']) for r in rows)
        cand=transform(rows);a=np.array([float(r['base']) for r in rows]);y=np.array([float(r['sub_temp']) for r in rows])
        # Adversarial future/day-isolation checks for each greenhouse and every cut hour in representative first day.
        for farm in ['F13','F47']:
            day=min(int(r['day']) for r in rows if r['farm']==farm)
            for hour in range(24):
                changed=[dict(r) for r in rows]
                for r in changed:
                    if r['farm']!=farm or int(r['day'])>day or (int(r['day'])==day and int(r['hour'])>hour):r['base']=str(float(r['base'])+1000);r['in_temp']='1000'
                altered=transform(changed);sel=np.array([r['farm']==farm and int(r['day'])==day and int(r['hour'])<=hour for r in rows]);assert np.array_equal(cand[sel],altered[sel])
        ea,eb=a-y,cand-y;ra=float(np.sqrt(np.mean(ea**2)));rb=float(np.sqrt(np.mean(eb**2)))
        assert abs(ra-math.sqrt(math.fsum(float(e)**2 for e in ea)/len(ea)))<1e-12
        assert abs(rb-math.sqrt(math.fsum(float(e)**2 for e in eb)/len(eb)))<1e-12
        # Independent arithmetic row-by-row, using a dictionary keyed by greenhouse/day/hour.
        prev={(r['farm'],int(r['day']),int(r['hour'])):r for r in rows};diff=0.
        for i,r in enumerate(rows):
            t=float(r['in_temp']) if r['in_temp'] else math.nan;p=prev.get((r['farm'],int(r['day']),int(r['hour'])-1));pt=float(p['in_temp']) if p and p['in_temp'] else math.nan
            q=float(r['base'])
            if p and math.isfinite(t) and math.isfinite(pt):
                g=max(0.,min(1.,(t-8)/2));q=math.fsum([q,-.2*g*(q-float(p['base'])),.1*g*(t-pt)])
            diff=max(diff,abs(q-cand[i]));r=dict(r);r['candidate']=str(cand[i]);record.append(r)
        assert diff<1e-12
        item=dict(validator=key[0],seed=key[1],context=key[2],n=len(rows),baseline_rmse=ra,candidate_rmse=rb,delta_pct=100*(rb/ra-1),independent_formula_maxdiff=diff)
        if key[0]=='DIAG10':
            blocks=collections.defaultdict(list)
            for r,x,z in zip(rows,ea,eb):blocks[r['farm']+'_'+str(int(r['day'])//5)].append(float(z*z-x*x))
            sums=np.array([math.fsum(blocks[k]) for k in sorted(blocks)]);counts=np.array([len(blocks[k]) for k in sorted(blocks)]);ix=rng.integers(0,len(sums),size=(20000,len(sums)));values=sums[ix].sum(1)/counts[ix].sum(1)
            item.update(p_worse=float(np.mean(values>=0)),ci99=np.quantile(values,[.005,.995]).tolist())
        summary.append(item);print(json.dumps(item),flush=True)
    verdict='PASS' if all(s['delta_pct']<0 for s in summary) and all(s['p_worse']<.005 and s['ci99'][1]<0 for s in summary if s['validator']=='DIAG10') else 'REJECT'
    data=dict(verdict=verdict,summary=summary,formula_audit='PASS',causality_cases=12*2*24,source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest())
    for target in [OUT/'result.json',HERE/'verification.json']:target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'oof.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(record[0]));writer.writeheader();writer.writerows(record)
    print('T-F1',verdict)
if __name__=='__main__':main()
