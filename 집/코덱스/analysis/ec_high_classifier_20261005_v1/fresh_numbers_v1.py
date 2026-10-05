from pathlib import Path
import csv,json,math
H=Path(__file__).resolve().parent
OUT=H.parents[3]/'집/코덱스/local'/H.name

def metric(rows,field,threshold):
    high=[r for r in rows if int(r['high'])];low=[r for r in rows if not int(r['high'])]
    selected=[r for r in rows if float(r[field])>=threshold(r)]
    tp=sum(int(r['high']) for r in selected);fp=len(selected)-tp
    pair=math.fsum(1 if float(a[field])>float(b[field]) else .5 if float(a[field])==float(b[field]) else 0 for a in high for b in low)/(len(high)*len(low))
    return dict(n=len(rows),high=len(high),tp=tp,fn=len(high)-tp,fp=fp,tn=len(low)-fp,recall=tp/len(high),precision=tp/len(selected) if selected else None,auc=pair,brier=math.fsum((float(r[field])-int(r['high']))**2 for r in rows)/len(rows))

def main():
    verified=json.loads((H/'verification_v1.json').read_text(encoding='utf-8'));result=[]
    for mode in ['PAST','ALL']:
        for s in [7,101,2024]:
            rows=[]
            for k in range(10):
                with (OUT/f'outer_{mode}_{k}_{s}.csv').open(encoding='utf-8',newline='') as f:rows.extend(csv.DictReader(f))
            assert len(rows)==8640 and len({r['row_id'] for r in rows})==8640
            for h in [0,6,12,23]:
                q=[r for r in rows if int(r['hour'])==h];assert len(q)==360
                model=metric(q,'p',lambda r:float(r['threshold']));base=metric(q,'prefix_A',lambda r:.9)
                expected=next(r for r in verified['results'] if r['mode']==mode and r['seed']==s and r['context']=='outer' and r['hour']==h)
                for key,got in [('model',model),('baseline',base)]:
                    for name,value in got.items():
                        target=expected[key][name]
                        assert value==target if value is None else abs(value-target)<1e-12,(key,name,value,target)
                result.append(dict(mode=mode,seed=s,hour=h,model=model,baseline=base))
    prep=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'))
    future={mode:sum(r['future_source_rows'] for r in prep['records'] if r['mode']==mode and not r['inner'] and r['s']==7) for mode in ['PAST','ALL']}
    payload=dict(status='PASS_STDLIB_COUNTS_PAIRWISE_AUC_BRIER',results=result,future_source_rows_outer_one_seed=future)
    with (H/'fresh_numbers_v1.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    print(json.dumps(payload,ensure_ascii=False))
if __name__=='__main__':main()
