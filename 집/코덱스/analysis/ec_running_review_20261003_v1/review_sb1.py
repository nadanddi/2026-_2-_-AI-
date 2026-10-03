from pathlib import Path
import sys,csv,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd,numpy as np
rows=[]
with open(ROOT/'집/클로드/research/local/ec3_SB1_all.csv',encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        if r['validator'] in ['DIAG10','A','B','EXT10','EXT12']:rows.append(r)
o=pd.DataFrame(rows);scores=[];checks=0
for v,g in o.groupby('validator'):
    y=g.sub_ec.astype(float).to_numpy()
    for seed in [7,101,2024]:
        a=g[f'r3s_{seed}'].astype(float).to_numpy();b=g[f'sb_{seed}'].astype(float).to_numpy()
        sp=g[f'sp_{seed}'].astype(float).to_numpy();flag=g.is2.astype(float).to_numpy()
        assert np.max(np.abs(b-np.where(flag==1,.5*a+.5*sp,a)))<1e-12;checks+=len(g)
        rb=math.sqrt(math.fsum((float(x)-float(p))**2 for x,p in zip(y,a))/len(g))
        rc=math.sqrt(math.fsum((float(x)-float(p))**2 for x,p in zip(y,b))/len(g))
        assert abs(rb-np.sqrt(np.mean((y-a)**2)))<1e-12
        assert abs(rc-np.sqrt(np.mean((y-b)**2)))<1e-12;checks+=2
        scores.append(dict(validator=v,seed=seed,n=len(g),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1)))
score=pd.DataFrame(scores);score.to_csv(H/'SB1_independent_scores_v1.csv',index=False)
(H/'SB1_review_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,public_rows=len(o),improved_cells=int((score.change_pct<0).sum()),limitations=['R3S baseline differs from current season_v2','No EL1 or final lock re-score','Specialist true role labels derive from full-day training weather; causal query flag is available','Rejection does not prove all specialists fail']),indent=2),encoding='utf-8')
print(score.to_string(index=False))
print('CHECKS',checks)
