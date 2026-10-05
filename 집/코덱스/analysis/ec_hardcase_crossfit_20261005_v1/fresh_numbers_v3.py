from pathlib import Path
import json,csv,math,random,statistics
H=Path(__file__).resolve().parent;OUT=H.parents[3]/'집/코덱스/local'/H.name

def counts(q,name):
    result=dict(n=len(q),high=0,tp=0,fn=0,fp=0,tn=0)
    for r in q:
        high=int(r['high']);sel=int(r[name]) if name!='baseline' else int(float(r['prefix_A'])>=.9);result['high']+=high;result['tp' if high and sel else 'fn' if high else 'fp' if sel else 'tn']+=1
    result['recall']=result['tp']/result['high'] if result['high'] else None;den=result['tp']+result['fp'];result['precision']=result['tp']/den if den else None
    return result

def main():
    verified=json.loads((H/'verification_v5.json').read_text(encoding='utf-8'));rc=json.loads((H/'receipt_v2.json').read_text(encoding='utf-8'));groups={};train=[]
    for rec in rc['files']:
        with (OUT/rec['path']).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        if rec['context']=='outer':groups.setdefault((rec['v'],rec['s'],rec['mode']),[]).extend(rows)
        elif rec['mode']=='UNIFORM':
            q=[r for r in rows if int(r['hour'])==23];train.append(dict(v=rec['v'],k=rec['k'],seed=rec['s'],n=len(q),high=sum(int(r['high']) for r in q),hard_high=sum(int(r['hard_high']) for r in q),hard_low=sum(int(r['hard_low']) for r in q)))
    result=[];changes=[]
    for (v,s,mode),rows in groups.items():
        if v=='DIAG10':assert len(rows)==8640 and len({r['row_id'] for r in rows})==8640
        for h in [0,6,12,23]:
            q=[r for r in rows if int(r['hour'])==h];base=counts(q,'baseline')
            for selection in ['DIRECT','GUARD']:
                got=counts(q,selection);expected=next(r for r in verified['results'] if r['v']==v and r['seed']==s and r['mode']==mode and r['context']=='outer' and r['hour']==h and r['selection']==selection)
                for name,value in got.items():assert value==expected['model'][name],(v,s,mode,h,name)
                for name,value in base.items():assert value==expected['baseline'][name]
                result.append(dict(v=v,seed=s,mode=mode,hour=h,selection=selection,model=got,baseline=base))
            if h==23:
                for r in q:
                    if int(r['GUARD'])!=int(float(r['prefix_A'])>=.9):changes.append(dict(v=v,seed=s,mode=mode,row_id=r['row_id'],high=int(r['high']),p=float(r['p']),prefix_A=float(r['prefix_A'])))
    assert train==verified['support']
    # A GUARD veto cannot add false positives by construction. Its bootstrap is diagnostic only.
    boots={}
    for mode in ['UNIFORM','HARD']:
        byday={}
        for s in [7,101,2024]:
            for r in groups['DIAG10',s,mode]:
                if int(r['hour'])!=23:continue
                y=int(r['high']);base=int(float(r['prefix_A'])>=.9);pick=int(r['GUARD']);byday.setdefault((r['farm'],int(r['day'])),[]).append((int(pick!=y)-int(base!=y),int(y==0)*(pick-base)))
        blocks=[]
        for farm in ['F13','F47']:
            days=sorted(k for k in byday if k[0]==farm)
            for start in range(0,len(days),5):
                chunk=days[start:start+5];blocks.append((math.fsum(math.fsum(a for a,b in byday[k])/3 for k in chunk),math.fsum(math.fsum(b for a,b in byday[k])/3 for k in chunk)))
        rng=random.Random(20241006);errors=[];fps=[]
        for i in range(10000):
            sampled=[blocks[rng.randrange(len(blocks))] for j in range(len(blocks))];errors.append(math.fsum(a for a,b in sampled));fps.append(math.fsum(b for a,b in sampled))
        errors.sort();fps.sort();boots[mode]=dict(blocks=len(blocks),replicates=10000,error_change=math.fsum(a for a,b in blocks),fp_change=math.fsum(b for a,b in blocks),error_ci95=[errors[249],errors[9749]],fp_ci95=[fps[249],fps[9749]],p_error_non_improvement=sum(a>=0 for a in errors)/10000,diagnostic_only=True)
    payload=dict(status='PASS_STDLIB_COUNTS_RATIOS_SUPPORT_BOOTSTRAP',results=result,support=train,changes=changes,bootstrap=boots)
    with (H/'fresh_numbers_v3.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':payload['status'],'bootstrap':boots},ensure_ascii=False))
if __name__=='__main__':main()

