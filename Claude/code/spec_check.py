import pandas as pd, numpy as np, os
s=pd.read_csv('submission.csv', dtype={'row_id':str}); t=pd.read_csv('정형데이터/test_X.csv', usecols=['row_id'], dtype={'row_id':str})
ok = list(s.columns)==['row_id','sub_temp','sub_ec'] and len(s)==1440 and s.row_id.is_unique and set(s.row_id)==set(t.row_id) and s.notna().all().all() and np.isfinite(s[['sub_temp','sub_ec']].values).all() and open('submission.csv','rb').read(3)!=b'\xef\xbb\xbf'
print('platform spec:', 'ALL PASS' if ok else 'FAIL')
