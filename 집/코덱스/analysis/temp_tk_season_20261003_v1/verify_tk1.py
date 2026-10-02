from world import *
import math
a=pd.read_csv(HERE/'TK1_error_map.csv');o=pd.read_csv(OUT/'TK1_predictions.csv');checks=[]
for member in ['BASE','CODEX','PFN','W30G','W30','G_C2']:
    for segment in ['early','late','late_calendar_early','late_calendar_late','sealed','F13_late','F47_late']:
        d=o[(o.validator=='DIAG10')&(o.member==member)&(o.base_seed==7)&(o.context=='1-8')]
        mask={'early':d.day<179,'late':d.day>=179,'late_calendar_early':(d.day>=179)&(d.cal<70),'late_calendar_late':(d.day>=179)&(d.cal>=70),'sealed':d.sealed,'F13_late':(d.day>=179)&(d.farm=='F13'),'F47_late':(d.day>=179)&(d.farm=='F47')}[segment];d=d[mask]
        r=a[(a.validator=='DIAG10')&(a.member==member)&(a.base_seed==7)&(a.context=='1-8')&(a.segment==segment)].iloc[0]
        e=[float(p)-float(y) for p,y in zip(d.prediction,d.sub_temp)];score=math.sqrt(math.fsum(v*v for v in e)/len(e));bias=math.fsum(e)/len(e);days={}
        for f,day,v in zip(d.farm,d.day,e):days.setdefault((f,int(day)),[]).append(v)
        level=math.fsum(math.fsum(vs)**2/len(vs) for vs in days.values())/math.fsum(v*v for v in e)
        assert max(abs(score-r.rmse),abs(bias-r.bias),abs(level-r.level_sse_fraction))<1e-12
        checks.append(dict(member=member,segment=segment,n=len(e),days=len(days),rmse=score,bias=bias,level_fraction=level))
(HERE/'TK1_verification.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2),encoding='utf-8');print('42 error map cells verified by math.fsum')
print(a.query("validator=='DIAG10' and base_seed==7 and context=='1-8' and segment in ['F13_late','F47_late','early','late']")[['member','segment','rmse','bias','level_sse_fraction']].to_string(index=False))
