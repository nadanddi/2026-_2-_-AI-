from pathlib import Path
import sys, math, json, hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
LOCAL=Path(env.LOCAL)
result={'scope':'Saved public OOF only, EL1 excluded. No fitted models, no weight selection. Directional derivatives are post-hoc diagnostic oracles.'}
for arm in ['HM1','HM2']:
    path=LOCAL/f'ec3_{arm}_all.csv'
    frame=pd.read_csv(path)
    frame=frame[frame.validator.ne('EL1')].copy()
    cells=[]
    for v,g in frame.groupby('validator'):
        for seed in [7,101,2024]:
            b=g[f'r3s_{seed}'].to_numpy()
            sp=g[f'sp_{seed}'].to_numpy()
            flag=g[f'flag_{seed}'].to_numpy(bool)
            e=b-g.sub_ec.to_numpy()
            direction=np.where(flag,sp-b,0.)
            cross=float(np.mean(e*direction)); square=float(np.mean(direction**2))
            cross_manual=math.fsum(float(a)*float(c) for a,c in zip(e,direction))/len(e)
            square_manual=math.fsum(float(c)**2 for c in direction)/len(e)
            assert abs(cross-cross_manual)<1e-13
            assert abs(square-square_manual)<1e-13
            delta=float(np.mean((g[f'hm_{seed}'].to_numpy()-g.sub_ec.to_numpy())**2-e**2))
            identity=2*.5*cross+.5**2*square
            assert abs(delta-identity)<1e-13
            assert np.max(np.abs(g[f'hm_{seed}'].to_numpy()-(b+.5*direction)))<1e-13
            cells.append(dict(validator=v,seed=seed,n=len(g),flagged=int(flag.sum()),mean_e_direction=cross,mean_direction_sq=square,derivative_at_zero=2*cross,observed_half_delta_mse=delta,quadratic_oracle_w=-cross/square if square else None))
    result[arm]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cells':cells}

# Re-check three reported quantities by independent scalar arithmetic; label-based groups are diagnostic only.
frame=pd.read_csv(LOCAL/'ec3_HM1_all.csv')
d=frame[frame.validator.eq('DIAG10')].copy()
high=d.groupby(['farm','day']).sub_ec.transform('mean').ge(1.)
flag=d.flag_7.astype(bool)
groups={}
for name,mask in [('true_high',high&flag),('false_high',~high&flag)]:
    g=d[mask]
    a=float(np.sum((g.r3s_7-g.sub_ec)**2)); c=float(np.sum((g.hm_7-g.sub_ec)**2))
    assert abs(a-math.fsum((float(p)-float(y))**2 for p,y in zip(g.r3s_7,g.sub_ec)))<1e-10
    assert abs(c-math.fsum((float(p)-float(y))**2 for p,y in zip(g.hm_7,g.sub_ec)))<1e-10
    groups[name]={'n':len(g),'baseline_sse':a,'candidate_sse':c}
result['verified_HM1_flag_groups']=groups
pr=pd.read_csv(LOCAL/'pr1_day_dist.csv')
coverage=float(((pr.y>=pr.q10)&(pr.y<=pr.q90)).mean())
manual=sum(float(y)>=float(lo) and float(y)<=float(hi) for y,lo,hi in zip(pr.y,pr.q10,pr.q90))/len(pr)
assert coverage==manual
result['PR1_saved_public_coverage']={'n':len(pr),'inside':int(round(coverage*len(pr))),'coverage':coverage,'nominal':.8}
(OUT/'direction_check_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for arm in ['HM1','HM2']:
    print(arm)
    for cell in result[arm]['cells']:
        print(cell['validator'],cell['seed'],'derivative',round(cell['derivative_at_zero'],7),'oracle_w',round(cell['quadratic_oracle_w'],5))
print(result['verified_HM1_flag_groups'])
print(result['PR1_saved_public_coverage'])
