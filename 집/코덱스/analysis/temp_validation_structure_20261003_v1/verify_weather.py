from pathlib import Path
import sys,csv,json,math,collections
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
with (Path(env.DATA)/'train_X.csv').open(encoding='utf-8-sig',newline='') as f:
    days=collections.defaultdict(list)
    for r in csv.DictReader(f):
        if r['row_id'][:3] in ['F13','F47']:days[(r['row_id'][:3],int(r['row_id'][4:7]))].append(r)
weather=['out_temp','out_hum','out_rad','out_wspd'];vectors=[]
for key,rows in sorted(days.items()):
    rows.sort(key=lambda r:r['row_id']);vectors.append(tuple(tuple(r[c] for r in rows) for c in weather))
counts=collections.Counter(vectors);pairs=sum(n*(n-1)//2 for n in counts.values());audit=json.loads((HERE/'audit.json').read_text(encoding='utf-8'))
assert pairs==audit['exact_weather_pairs']==438
with (HERE/'weather_groups.csv').open(encoding='utf-8',newline='') as f:groups={(r['farm'],int(r['day'])):int(r['weather_group']) for r in csv.DictReader(f)}
keys=sorted(days);exposed=0
for k in range(10):
    va=[key for farm in ['F13','F47'] for i,key in enumerate([q for q in keys if q[0]==farm]) if (i//5)%10==k];forbidden={(f,d+j) for f,d in va for j in [-1,0,1]};tg={groups[key] for key in keys if key not in forbidden};n=sum(groups[key] in tg for key in va);assert n==audit['folds'][k]['validation_weather_shared_train'];exposed+=n
assert len(keys)==audit['days']==400 and exposed==291
answer=dict(status='PASS',method='CSV exact string vector hashing, explicit independent day exclusion sets',days=len(keys),exact_weather_pairs=pairs,validation_weather_shared_train=exposed)
(HERE/'verification.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(answer))
