from pathlib import Path
import sys,csv,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd,numpy as np
rows=[]
with open(ROOT/'집/클로드/research/local/ec3_ST5_all.csv',encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        if r['validator'] in ['DIAG10','A','B','EXT10','EXT12']:rows.append(r)
o=pd.DataFrame(rows);scores=[];checks=0
for v,g in o.groupby('validator'):
    y=g.sub_ec.astype(float).to_numpy()
    for seed in [7,101,2024]:
        a=g[f'r3s_{seed}'].astype(float).to_numpy();b=g[f'st5_{seed}'].astype(float).to_numpy()
        rb=math.sqrt(math.fsum((float(x)-float(p))**2 for x,p in zip(y,a))/len(g));rc=math.sqrt(math.fsum((float(x)-float(p))**2 for x,p in zip(y,b))/len(g))
        assert abs(rb-np.sqrt(np.mean((y-a)**2)))<1e-12;assert abs(rc-np.sqrt(np.mean((y-b)**2)))<1e-12;checks+=2
        scores.append(dict(validator=v,seed=seed,n=len(g),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1)))
score=pd.DataFrame(scores);score.to_csv(H/'ST5_independent_scores_v1.csv',index=False)
(H/'ST5_review_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,public_rows=len(o),limitations=['ST5 weather prefix flag is available under actual MASK; no classifier fitting in this experiment','R3S baseline differs from current season_v2','No EL1 or final lock re-score']),indent=2),encoding='utf-8')
print(score.to_string(index=False))
