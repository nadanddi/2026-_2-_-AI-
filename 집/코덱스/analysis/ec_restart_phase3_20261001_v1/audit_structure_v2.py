"""Correct day-occurrence reporting; no changes to model folds/predictions."""
import run_benchmark as b
import pandas as pd
import json

def main():
    _,_,lab,_,_,fds=b.prepare()
    r=json.loads((b.HERE/'structure_full_audit.json').read_text(encoding='utf-8'))
    days=lab[['farm','day']].drop_duplicates()
    records=[]
    for name,i,vd in fds:
        records.extend((name,i,f,int(day)) for f,day in days.itertuples(index=False,name=None) if (f,int(day)) in vd)
    a=pd.DataFrame(records,columns=['validator','fold','farm','day'])
    counts=a.groupby(['validator','farm','day']).size()
    r['validation_overlap']={name:{'day_occurrences':int(g.sum()),'unique_days':len(g),'days_repeated':int(g.gt(1).sum()),'max_occurrences':int(g.max())} for name,g in counts.groupby(level=0)}
    assert r['validation_overlap']['DIAG10']=={'day_occurrences':360,'unique_days':360,'days_repeated':0,'max_occurrences':1}
    for name,g in counts.groupby(level=0):
        # Independent Python sets and list counts.
        rr=[(f,d) for n,i,f,d in records if n==name]
        assert len(rr)==int(g.sum()) and len(set(rr))==len(g)
        assert sum(rr.count(k)>1 for k in set(rr))==int(g.gt(1).sum())
    r['correction']='v1 counted each of the 24 hourly rows as day occurrence; v2 uses distinct farm-day per fold. Model training and fold fingerprint counts unaffected.'
    r['independent_overlap_verification']='PASS'
    b.save(b.HERE/'structure_full_audit_verified.json',r)
    print(json.dumps(r['validation_overlap'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
