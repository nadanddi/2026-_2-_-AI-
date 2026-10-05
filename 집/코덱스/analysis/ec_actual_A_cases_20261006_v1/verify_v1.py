from pathlib import Path
import csv,json,math,hashlib,collections
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
def read(name):
    with (OUT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(name,obj):
    with (H/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def key(r):return (r['v'],int(r['k']),int(r['s']),r['farm'],int(r['day']))
def ckey(r):return key(r)+(int(r['hour']),)
def main():
    result=json.loads((H/'result_v1.json').read_text(encoding='utf-8'))
    for z in result['files']:assert hashlib.sha256((OUT/z['path']).read_bytes()).hexdigest()==z['sha']
    rows=read('reconstructed_rows_v1.csv');cases=read('case_registry_v1.csv');cons=read('seed_consensus_v1.csv')
    groups=collections.defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    gaps=dict(A=0.,prefix_A=0.,prefix_proxy=0.,y_day=0.);expected={}
    for k,g in groups.items():
        g.sort(key=lambda r:int(r['hour']));assert [int(r['hour']) for r in g]==list(range(24))
        assert len({r['row_id'] for r in g})==24
        yy=math.fsum(float(r['y']) for r in g)/24;hist=[];ah=[];ph=[]
        for r in g:
            raw=.8*float(r['r3'])+.2*float(r['pfn']);hist.append(raw)
            a=min(float(r['hi']),max(float(r['lo']),.5*raw+.5*math.fsum(hist)/len(hist)))
            gaps['A']=max(gaps['A'],abs(a-float(r['A'])))
            ah.append(float(r['A']));ph.append(float(r['proxy']))
            scores={'A':math.fsum(ah)/len(ah),'proxy':math.fsum(ph)/len(ph)}
            for source,x in scores.items():gaps['prefix_'+source]=max(gaps['prefix_'+source],abs(x-float(r['prefix_'+source])))
            if int(r['hour']) in [0,6,12,23]:expected[ckey(r)]=(yy,scores)
    assert len(expected)==len(cases)==len(groups)*4
    aggregates=collections.defaultdict(collections.Counter);cgroups=collections.defaultdict(list)
    sets=collections.defaultdict(lambda:dict(A=set(),proxy=set()))
    for r in cases:
        yy,scores=expected.pop(ckey(r));high=int(yy>=1);assert high==int(r['high']);gaps['y_day']=max(gaps['y_day'],abs(yy-float(r['y_day'])))
        for source,x in scores.items():
            flags=dict(selected=int(x>=.9),hard_high=int(high and x<1.2),missed_high=int(high and x<.9),hard_low=int(not high and x>=.9))
            for label,val in flags.items():assert int(r[source+'_'+label])==val,(ckey(r),source,label)
            z=aggregates[(r['v'],int(r['s']),int(r['hour']),source)]
            z.update(n=1,high=high,tp=high*flags['selected'],fn=flags['missed_high'],fp=flags['hard_low'],hard_high=flags['hard_high'],captured_low_high=flags['hard_high']*flags['selected'],ordinary=1-high)
            for label in ['hard_high','hard_low']:
                if flags[label]:sets[(r['v'],int(r['s']),int(r['hour']),label)][source].add((int(r['k']),r['farm'],int(r['day'])))
        cgroups[(r['v'],int(r['k']),r['farm'],int(r['day']),int(r['hour']))].append(r)
    assert not expected
    for z in result['summary']:
        got=aggregates[z['v'],z['s'],z['hour'],z['source']]
        assert all(z[k]==v for k,v in got.items()),(z,got)
    stable=[]
    for r in cons:
        g=cgroups.pop((r['v'],int(r['k']),r['farm'],int(r['day']),int(r['hour'])));assert sorted(int(x['s']) for x in g)==[7,101,2024]
        for label,field in [('A_hard_high_votes','A_hard_high'),('A_hard_low_votes','A_hard_low'),('A_missed_votes','A_missed_high'),('proxy_hard_high_votes','proxy_hard_high'),('proxy_hard_low_votes','proxy_hard_low')]:assert int(r[label])==sum(int(x[field]) for x in g)
        for which,fun in [('min',min),('max',max)]:assert abs(float(r['A_prefix_'+which])-fun(float(x['prefix_A']) for x in g))<1e-12
        if r['v']=='DIAG10' and int(r['hour'])==23 and (int(r['A_hard_high_votes']) or int(r['A_hard_low_votes'])):stable.append(r)
    assert not cgroups and max(gaps.values())<1e-12,gaps
    overlap=[]
    for (v,s,h,label),z in sorted(sets.items()):
        a,p=z['A'],z['proxy'];overlap.append(dict(v=v,s=s,hour=h,label=label,A=len(a),proxy=len(p),both=len(a&p),A_only=len(a-p),proxy_only=len(p-a),jaccard=len(a&p)/len(a|p) if a|p else 1.,A_only_ids=sorted(a-p),proxy_only_ids=sorted(p-a)))
    proof=dict(status='PASS',method='stdlib CSV + scalar math.fsum + manual integer counts + set intersection + seed votes',rows=len(rows),case_rows=len(cases),consensus_rows=len(cons),maxdiff=gaps,summary=result['summary'],overlap=overlap,DIAG23_cases=stable,limitations=['Saved original OOF components reconstructed; no fresh fit','Repeated public validation, not an untouched holdout','Whole-day labels retrospective; hourly scores use prefix only','No classifier training or RMSE improvement measured','Global OOF cannot be directly split to validate a new classifier'])
    save('verification_v1.json',proof)
    print(json.dumps({k:proof[k] for k in ['status','rows','case_rows','consensus_rows','maxdiff']},ensure_ascii=False))
    for z in result['summary']:
        if z['v']=='DIAG10' and z['hour']==23:print(json.dumps(z,ensure_ascii=False))
    for z in overlap:
        if z['v']=='DIAG10' and z['hour']==23:print(json.dumps(z,ensure_ascii=False))
if __name__=='__main__':main()
