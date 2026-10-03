"""Post-hoc failure diagnosis, never used to select a new blend weight."""
from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd,numpy as np
from scipy.stats import spearmanr
o=pd.read_csv(ROOT/'집/코덱스/local/source_history_20261003_v1/oof.csv',float_precision='round_trip')
d=o[(o.validator=='DIAG10')&(o.seed==7)].copy();d['gap']=np.where(d.last_day>=0,d.day-d.last_day,np.nan)
d['residual']=d.y-d.baseline;d['correction']=d.candidate-d.baseline
daily=d.groupby(['farm','day']).agg(residual=('residual','mean'),correction=('correction','mean'),gap=('gap','first'),history_n=('history_n','first'))
rows=[]
for farm in ['all','F13','F47']:
    for period in ['all','early','late']:
        for band in ['all','no_history','gap1_3','gap4_6','gap7_12']:
            m=np.ones(len(d),bool)
            if farm!='all':m &= d.farm.eq(farm).to_numpy()
            if period!='all':m &= (d.day.ge(179) if period=='late' else d.day.lt(179)).to_numpy()
            if band=='no_history':m &= d.history_n.eq(0).to_numpy()
            if band.startswith('gap'):
                lo,hi=map(int,band[3:].split('_'));m &= d.gap.between(lo,hi).to_numpy()
            g=d[m];keys=set(zip(g.farm,g.day));dg=daily.loc[[k in keys for k in daily.index]]
            if len(g)==0:continue
            rb=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g.baseline))/len(g))
            rc=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g.candidate))/len(g))
            grad=math.fsum(float(e)*float(c) for e,c in zip(g.residual,g.correction))
            sq=math.fsum(float(c)**2 for c in g.correction)
            delta=math.fsum((float(y)-float(p))**2-(float(y)-float(b))**2 for y,p,b in zip(g.y,g.candidate,g.baseline))
            assert abs(delta-(sq-2*grad))<1e-10
            rho=float(spearmanr(dg.residual,dg.correction).statistic) if len(dg)>5 and dg.correction.std()>0 else None
            rows.append(dict(farm=farm,period=period,band=band,n_rows=len(g),n_days=len(dg),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1),residual_correction_rho=rho,gradient_dot=grad,correction_sse=sq,delta_sse=delta))
pd.DataFrame(rows).to_csv(H/'history_gap_diagnostic_v1.csv',index=False)
out=pd.DataFrame(rows);print(out[(out.farm=='all')&(out.period=='all')].to_string(index=False))
(H/'history_diagnostic_verification_v1.json').write_text(json.dumps(dict(status='PASS',sse_identity_checks=len(rows),n_days=len(daily),warning='post-hoc bands; no gate or weight tuning; only one seed diagnostic'),ensure_ascii=False,indent=2),encoding='utf-8')
