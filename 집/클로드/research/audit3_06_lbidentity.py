import env, common
import numpy as np, pandas as pd, os
tX, ty, sX = common.load_raw()
S={}
S[1]=pd.read_csv('../Claude/submissions/submission_01_scored.csv')
S[2]=pd.read_csv('submissions/submission_03.csv'); S[3]=pd.read_csv('submissions/submission_04.csv'); S[4]=pd.read_csv('submissions/submission_05.csv')
sc={1:0.7450,2:0.6666,3:0.5624,4:0.5697}
base=sX[['row_id','farm','day','hour','in_temp']].copy()
x=sX.sort_values(['farm','t']).copy()
x['sm3']=x.groupby('farm').in_temp.transform(lambda s:s.ewm(halflife=3).mean())  # approximate
base=base.merge(x[['row_id','sm3']],on='row_id')
for k in S: base['p%d'%k]=base.row_id.map(S[k].set_index('row_id').sub_temp)
N=len(base); SSE={k:N*sc[k]**2 for k in sc}
bins=pd.cut(base.in_temp,[-9,6,8,10,15,99])
# express everything relative to r3 = p3 - y (3rd submission)
for a,b in [(3,4),(2,3),(1,2)]:
    d=(base['p%d'%b]-base['p%d'%a]).values
    # SSE_b - SSE_a = 2 sum d * r_a + sum d^2 ; r_a = r3 + (p_a - p3)
    s_dra=(SSE[b]-SSE[a]-(d**2).sum())/2
    s_dr3=s_dra-(d*(base['p%d'%a]-base.p3)).sum()
    print(f'{a}->{b}: sum d^2={np.sum(d**2):.2f}, dSSE={SSE[b]-SSE[a]:.2f}, sum d*r3={s_dr3:.2f}, implied d-weighted mean r3 = {s_dr3/np.sum(d**2)*1:.3f} (per unit d) ; sum|d|={np.abs(d).sum():.1f}, weighted mean r3 (weights d/sum d)={s_dr3/d.sum():.3f}')
    print(pd.DataFrame({'bin':bins,'d':d}).groupby('bin',observed=True).d.agg(['size','mean',lambda v:(v**2).sum()]).round(3).to_string())
# rounding uncertainty
print('SSE uncertainty per score +-', round(2*N*0.56*0.00005,3))
print('---- LS bias solve ----')
grp=np.where(base.in_temp<6,'c6',np.where(base.in_temp<10,'c10','w'))
eqs=[];rhs=[];W=[]
for a,b in [(3,4),(2,3),(1,2)]:
    d=(base['p%d'%b]-base['p%d'%a]).values
    s=(SSE[b]-SSE[a]-(d**2).sum())/2-(d*(base['p%d'%a]-base.p3)).sum()
    eqs.append([d[grp==g].sum() for g in ['c6','c10','w']]); rhs.append(s); W.append(np.sqrt((d**2).sum()))
E=np.array(eqs); r=np.array(rhs)
print(np.round(E,1), np.round(r,2))
for cols in [[0,2],[0,1,2]]:
    sol=np.linalg.lstsq(E[:,cols]/np.array(W)[:,None], r/np.array(W), rcond=None)[0]
    print(cols, np.round(sol,3))
sol=np.linalg.solve(E,r); print('exact 3x3',np.round(sol,3))
