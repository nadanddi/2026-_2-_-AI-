from pathlib import Path
import sys,math,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
O=ROOT/'집/코덱스/local/ec_tree_median_20261003_v1';z=dict(np.load(O/'first_fold_tree_predictions.npz'));d=pd.read_csv(O/'DIAG10_0_7.csv',float_precision='round_trip');assert np.array_equal(d.row_id,z['row_id']);checks=1;meddiff=meandiff=0
for j in range(len(d)):
 vals=sorted(float(x) for x in z['tree_predictions'][:,j]);n=len(vals);assert n==600;m=(vals[n//2-1]+vals[n//2])/2;avg=math.fsum(vals)/n;meddiff=max(meddiff,abs(m-z['raw_median'][j]));meandiff=max(meandiff,abs(avg-z['raw_mean'][j]));assert abs(m-float(d.new_et_raw.iloc[j]))<1e-12;checks+=2
assert meddiff<1e-12 and meandiff<1e-12
(H/'first_tree_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,rows=len(d),trees=600,median_maxdiff=meddiff,mean_maxdiff=meandiff,warning='first fold computation audit only; not a full performance decision'),indent=2),encoding='utf-8');print('PASS',checks)
