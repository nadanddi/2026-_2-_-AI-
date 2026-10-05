from pathlib import Path
import json,csv,math
H=Path(__file__).resolve().parent;OUT=H.parents[3]/'집/코덱스/local'/H.name

def count(rows,mode):
    got=dict(n=len(rows),high=0,tp=0,fn=0,fp=0,tn=0)
    for r in rows:
        high=int(r['high']);chosen=int(r[mode]) if mode!='baseline' else int(float(r['prefix_A'])>=.9);got['high']+=high
        got['tp' if high and chosen else 'fn' if high else 'fp' if chosen else 'tn']+=1
    got['recall']=got['tp']/got['high'] if got['high'] else None;got['precision']=got['tp']/(got['tp']+got['fp']) if got['tp']+got['fp'] else None
    return got

def main():
    verified=json.loads((H/'verification_v2.json').read_text(encoding='utf-8'));rc=json.loads((H/'receipt_v1.json').read_text(encoding='utf-8'));groups={};changes=[];calibration=[]
    for rec in rc['manifest']:
        if rec['context']!='outer':continue
        with (OUT/rec['path']).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        groups.setdefault((rec['v'],rec['s']),[]).extend(rows)
        b=json.loads((OUT/rec['model']).read_text(encoding='utf-8'));calibration.append(dict(v=rec['v'],k=rec['k'],seed=rec['s'],fallback=b['fallback'],split_counts=[z['counts'] for z in b['split']],cuts=b['cuts']))
        for r in rows:
            if int(r['hour'])!=23:continue
            if int(r['GUARD'])!=int(float(r['prefix_A'])>=.9):changes.append(dict(v=rec['v'],seed=rec['s'],row_id=r['row_id'],high=int(r['high']),p=float(r['p']),prefix_A=float(r['prefix_A']),threshold=b['cuts']['GUARD']))
    result=[]
    for (v,s),rows in groups.items():
        assert len({r['row_id'] for r in rows})==len(rows)
        for h in [0,6,12,23]:
            q=[r for r in rows if int(r['hour'])==h];base=count(q,'baseline')
            for mode in ['SPLIT','GUARD']:
                got=count(q,mode);expected=next(r for r in verified['results'] if r['v']==v and r['seed']==s and r['context']=='outer' and r['hour']==h and r['mode']==mode)
                for name,value in got.items():assert value==expected['model'][name]
                for name,value in base.items():assert value==expected['baseline'][name]
                result.append(dict(v=v,seed=s,hour=h,mode=mode,model=got,baseline=base))
    payload=dict(status='PASS_STDLIB_RAW_COUNTS_RATIOS',results=result,guard_changes=changes,calibration=calibration)
    with (H/'fresh_numbers_v2.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':payload['status'],'guard_change_count':len(changes),'guard_lost_high':sum(r['high'] for r in changes)},ensure_ascii=False))
if __name__=='__main__':main()

