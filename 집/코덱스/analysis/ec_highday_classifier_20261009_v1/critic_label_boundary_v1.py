from pathlib import Path
import csv,json,math,decimal
import runtime_v1
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[3];path=R/'공용/대회자료/정형데이터/참가자_배포/train_y.csv';exact={}
for q in csv.DictReader(path.open(encoding='utf8')):
 farm,d,h=q['row_id'].split('_')
 if farm in ['F13','F47']:exact.setdefault((farm,int(d)),[]).append(q['sub_ec'])
means={k:sum(decimal.Decimal(v) for v in a)/decimal.Decimal(24) for k,a in exact.items()};floats={k:math.fsum(float(v) for v in a)/24 for k,a in exact.items()}
comparisons={}
for name,precision in [('pandas_default',None),('pandas_round_trip','round_trip')]:
 df=pd.read_csv(path,usecols=['row_id','sub_ec'],float_precision=precision);z=df.row_id.str.split('_',expand=True);df['farm']=z[0];df['day']=z[1].astype(int);df=df[df.farm.isin(['F13','F47'])];g=df.groupby(['farm','day']).sub_ec.mean();comparisons[name]=g.to_dict()
border=[]
for k,v in means.items():
 if abs(v-decimal.Decimal('1.2'))<decimal.Decimal('1e-12'):border.append({'farm':k[0],'day':k[1],'decimal_mean':str(v),'fsum_mean':repr(floats[k]),'pandas_default_mean':repr(comparisons['pandas_default'][k]),'pandas_round_trip_mean':repr(comparisons['pandas_round_trip'][k])})
out={'threshold':'>=1.2','decimal_high_days':sum(v>=decimal.Decimal('1.2') for v in means.values()),'fsum_high_days':sum(v>=1.2 for v in floats.values()),'pandas_default_high_days':sum(v>=1.2 for v in comparisons['pandas_default'].values()),'pandas_round_trip_high_days':sum(v>=1.2 for v in comparisons['pandas_round_trip'].values()),'decimal_exact_equal_days':sum(v==decimal.Decimal('1.2') for v in means.values()),'borderline':border,'extra40_predictions_or_performance':False}
p=H/'critic_label_boundary_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
