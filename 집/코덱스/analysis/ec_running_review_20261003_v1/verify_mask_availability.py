from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import pandas as pd
W=['out_temp','out_hum','out_rad','out_wspd'];x=pd.read_csv(Path(env.DATA)/'test_X.csv',usecols=['row_id']+W)
assert len(x)==1440 and x.row_id.is_unique
counts={c:int(x[c].notna().sum()) for c in W};manual={c:sum(pd.notna(v) for v in x[c]) for c in W};assert counts==manual
result=dict(status='PASS',rows=len(x),nonmissing=counts,allowed_if_all_present=all(n==len(x) for n in counts.values()),scope='MASK missingness only; no values, distributions, frequencies, or parameters used for model fit; no targets read')
(H/'weather_MASK_availability_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
