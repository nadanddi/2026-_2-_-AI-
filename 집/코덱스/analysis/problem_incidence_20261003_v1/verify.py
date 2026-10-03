from diagnose import *

def main():
    t,e=load();frames={'TEMP':t,'EC':e};r=json.loads((HERE/'result.json').read_text(encoding='utf-8'));cache={};checks=0;empty=0
    for s in r['records']:
        key=(s['target'],s['validator'],s['seed'],s['context'])
        if key not in cache:
            q=frames[s['target']];q=q[(q.validator==s['validator'])&(q.seed==s['seed'])&(q.context==s['context'])]
            groups={}
            for row in q.itertuples():groups.setdefault((row.farm,int(row.day)),[]).append(row)
            cache[key]=groups
        groups=cache[key];scope=s['segment']
        groups={k:rows for k,rows in groups.items() if scope not in ['F13','F47','early','late'] or (k[0]==scope if scope in ['F13','F47'] else (k[1]<179 if scope=='early' else k[1]>=179))}
        if not groups:
            assert s['days']==0 and s['event_days']==0
            empty+=1;continue
        n=0;total=[];event=[];failed=[];good=0;events=0;faildays=0;eventrows=0;missing=0
        for (farm,day),rows in groups.items():
            y=math.fsum(float(a.y) for a in rows)/len(rows);p=math.fsum(float(a.pred) for a in rows)/len(rows)
            err=[(float(a.pred)-float(a.y))**2 for a in rows];rmse=math.sqrt(math.fsum(err)/len(err));bias=p-y;n+=len(rows);total.extend(err)
            if s['target']=='EC':occur=y>=s['threshold'];failure=occur and bias<=-.2;ok=occur and rmse<=.1
            else:
                air=[float(a.in_temp) for a in rows if pd.notna(a.in_temp)];gap=y-math.fsum(air)/len(air) if air else float('nan');missing+=len(air)!=24
                occur=len(air)==24 and abs(gap)>=s['threshold']
                if scope=='warmer':occur=occur and gap>=2
                if scope=='cooler':occur=occur and gap<=-2
                failure=occur and bias*gap<0 and abs(bias)>=.5;ok=occur and rmse<=.5
            if occur:events+=1;eventrows+=len(rows);event.extend(err)
            if failure:faildays+=1;failed.extend(err)
            good+=ok
        actual=dict(days=len(groups),rows=n,event_days=events,event_pct=100*events/len(groups),event_rows=eventrows,event_sse_pct=100*math.fsum(event)/math.fsum(total),failure_days=faildays,failure_pct_all=100*faildays/len(groups),failure_sse_pct=100*math.fsum(failed)/math.fsum(total),good_days=good,good_pct_event=100*good/events if events else None,event_rmse=math.sqrt(math.fsum(event)/eventrows) if eventrows else None)
        if s['target']=='TEMP':actual['incomplete_air_days']=missing
        for field,v in actual.items():
            if v is None:assert s[field] is None
            else:assert abs(v-s[field])<1e-10,(key,scope,field,v,s[field])
        checks+=1
    for ex in r['examples']:
        context='1-8' if ex['target']=='TEMP' else '1-4';rows=cache[(ex['target'],'DIAG10',7,context)][(ex['farm'],ex['day'])]
        for name,attr in [('y','y'),('pred','pred')]:assert abs(math.fsum(float(getattr(a,attr)) for a in rows)/len(rows)-ex[name])<1e-12
        rmse=math.sqrt(math.fsum((float(a.pred)-float(a.y))**2 for a in rows)/len(rows));assert abs(rmse-ex['rmse'])<1e-12
    primary=daily(t[(t.validator=='DIAG10')&(t.seed==7)&(t.context=='1-8')],'TEMP')
    incomplete=primary[primary.air_n!=24].reset_index().to_dict(orient='records')
    partial_events=primary[(primary.gap.abs()>=2)&(primary.air_n>0)]
    result=dict(status='PASS',checked_nonempty_cells=checks,empty_scopes_not_interpreted=empty,example_checks=len(r['examples']),method='independent raw-row dictionaries/math.fsum versus pandas groupby',incomplete_air_days=incomplete,partial_air_sensitivity=dict(event_days=len(partial_events),event_sse_pct=100*partial_events.sse.sum()/primary.sse.sum()),result_hash=hashlib.sha256((HERE/'result.json').read_bytes()).hexdigest())
    (HERE/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='incomplete_air_days'}))

if __name__=='__main__':main()
