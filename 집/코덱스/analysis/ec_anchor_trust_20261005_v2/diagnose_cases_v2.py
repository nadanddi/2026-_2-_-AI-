"""Post-experiment error diagnosis, no ID exception or parameter changes."""
from pathlib import Path
import sys,csv,json,math,collections,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
def main():
 out={};cases=[('F13',231),('F13',233),('F47',216)]
 for mode in ['GATE','RIDGE']:
  v=json.loads((H/f'verification_{mode}_v1.json').read_text(encoding='utf-8'));assert v['status']=='PASS_ARITHMETIC_REPLAY';p=ROOT/'집/코덱스/local'/H.name/mode/'oof.csv';assert hashlib.sha256(p.read_bytes()).hexdigest()==v['aggregate'];days=collections.defaultdict(list)
  with p.open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):
    if r['validator']!='DIAG10':continue
    days[r['farm'],int(r['day']),int(r['seed'])].append(r)
  records=[]
  for (farm,day,seed),rs in days.items():
   n=len(rs);assert n==24;y=[float(r['y']) for r in rs];b=[float(r['baseline']) for r in rs];c=[float(r['candidate']) for r in rs];anchor=[float(r['anchor']) for r in rs];delta=[float(r['delta']) for r in rs];conf=[float(r['confidence']) for r in rs]
   records.append(dict(negative_changed_hours=sum(z<0 for z in delta),positive_changed_hours=sum(z>0 for z in delta),farm=farm,day=day,seed=seed,ymean=math.fsum(y)/24,baseline_mean=math.fsum(b)/24,candidate_mean=math.fsum(c)/24,anchor_mean=math.fsum(anchor)/24,confidence_mean=math.fsum(conf)/24,changed_hours=sum(abs(z)>0 for z in delta),delta_sse=math.fsum((a-t)**2-(z-t)**2 for a,z,t in zip(c,b,y)),baseline_sse=math.fsum((z-t)**2 for z,t in zip(b,y)),candidate_sse=math.fsum((a-t)**2 for a,t in zip(c,y))))
  out[mode]=dict(direction_by_segment={seg:{'changed_hours':sum(r['changed_hours'] for r in records if (r['ymean']>=1)==(seg=='high')),'negative_hours':sum(r['negative_changed_hours'] for r in records if (r['ymean']>=1)==(seg=='high')),'positive_hours':sum(r['positive_changed_hours'] for r in records if (r['ymean']>=1)==(seg=='high'))} for seg in ['high','ordinary']},selected_cases=[r for r in records if (r['farm'],r['day']) in cases],worst_added_loss=sorted(records,key=lambda r:-r['delta_sse'])[:12],most_improved=sorted(records,key=lambda r:r['delta_sse'])[:12],changed_days_by_seed={s:sum(r['seed']==s and r['changed_hours']>0 for r in records) for s in [7,101,2024]})
 with (H/'case_diagnosis_v2.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v['selected_cases'] for k,v in out.items()},ensure_ascii=False))
if __name__=='__main__':main()
