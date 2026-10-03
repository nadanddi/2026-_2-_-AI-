from pathlib import Path
import math,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
source=ROOT/'집/클로드/research/local/ec3_HM1_all.csv'
cached=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
hm=pd.concat([chunk[chunk.validator.ne('EL1')] for chunk in pd.read_csv(source,chunksize=2048)],ignore_index=True)
base=pd.read_csv(cached,float_precision='round_trip')
assert not base.validator.eq('EL1').any()
cells=[]
for seed in [7,101,2024]:
    d=hm.merge(base[base.seed.eq(seed)][['validator','validation_fold','row_id','sub_ec','season_v2']],on=['validator','validation_fold','row_id'],validate='one_to_one',suffixes=['','_base'])
    assert len(d)==len(hm)
    assert np.max(np.abs(d.sub_ec-d.sub_ec_base))<1e-15
    d['direction']=np.where(d[f'flag_{seed}'].to_numpy(bool),d[f'sp_{seed}']-d[f'r3s_{seed}'],0.)
    for v,g in d.groupby('validator'):
        for part,mask in [('all',np.ones(len(g),bool)),('late',g.day.ge(179).to_numpy())]:
            z=g[mask]
            if z.empty:continue
            e=z.season_v2.to_numpy()-z.sub_ec.to_numpy(); direction=z.direction.to_numpy()
            cross=float(np.mean(e*direction)); sq=float(np.mean(direction**2))
            manual=math.fsum(float(x)*float(t) for x,t in zip(e,direction))/len(z)
            assert abs(cross-manual)<1e-13
            square_manual=math.fsum(float(t)**2 for t in direction)/len(z)
            assert abs(sq-square_manual)<1e-13
            cells.append(dict(validator=v,part=part,seed=seed,n=len(z),flagged=int(z[f'flag_{seed}'].sum()),mean_e_direction=cross,mean_direction_sq=sq,derivative_at_zero=2*cross,quadratic_oracle_w=-cross/sq if sq else None))
result=dict(scope='Post-hoc derivative only, original HM1 direction held fixed; no candidate weight chosen, no score/adoption.',hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,cached]},key_and_label_match='PASS',cells=cells)
(OUT/'actual_baseline_direction_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for cell in cells:
    print(cell['validator'],cell['part'],cell['seed'],'derivative',round(cell['derivative_at_zero'],7),'oracle_w',round(cell['quadratic_oracle_w'],5) if cell['quadratic_oracle_w'] is not None else None)
