import env
import numpy as np, pandas as pd
S=pd.read_csv('local/audit3_10_daystats.csv').set_index(['farm','day'])
tr=S[S.grp.isin(['Q1-3','Q4'])]
c=tr.T_mean<11
print('cold train days',c.sum(),'of which Q4',(c&tr.grp.eq('Q4')).sum(), ' Q4 share %.2f vs overall %.2f'%((tr.grp.eq('Q4')[c]).mean(),tr.grp.eq('Q4').mean()))
print('Q4 cold days:',tr[c & tr.grp.eq('Q4')].index.tolist())
print('train days with sm3<8 rows: F13 114,154,157,158,170 F47 133,155,158,165,167,169 ->', {k:tr.grp.get(k) for k in [('F13',114),('F13',154),('F13',157),('F13',158),('F13',170),('F47',133),('F47',155),('F47',158),('F47',165),('F47',167),('F47',169)]})
