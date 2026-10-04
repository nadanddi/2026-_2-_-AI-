from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
with np.load(ROOT/'집/코덱스/local/ec_gpu_fixed_batch_audit_20261003_v1/fold1_context1_fixed.npz') as z:gpu={k:z[k] for k in z.files}
with np.load(ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1/DIAG10_1_pfn_1.npz') as z:cpu={k:z[k] for k in z.files}
assert np.array_equal(gpu['row_id'],cpu['row_id']) and np.array_equal(gpu['context_row_id'],cpu['context_row_id'])
delta=gpu['prediction']-cpu['prediction'];manual=max(abs(float(a)-float(b)) for a,b in zip(gpu['prediction'],cpu['prediction']));vector=float(np.max(np.abs(delta)));assert manual==vector
rm=math.sqrt(math.fsum(float(x)**2 for x in delta)/len(delta));assert abs(rm-float(np.sqrt(np.mean(delta**2))))<1e-12
r=json.loads((H/'result_v1.json').read_text());assert vector==r['cpu_raw_maxdiff'] and len(delta)==r['query_rows'];assert r['query_invariance_pass']==(r['repeat_maxdiff']==0 and all(c['bit_equal'] and c['maxdiff']==0 for c in r['checks']))
q=dict(status='PASS',checks=7,query_rows=len(delta),context_rows=len(gpu['context_row_id']),cpu_raw_maxdiff=manual,cpu_raw_rms=rm,cpu_equivalence_pass=manual<=1e-5,scope='saved query/context identity and CPU/GPU error independent scalar/NumPy; original two-fit query audit not rerun')
(H/'saved_verification_v1.json').write_text(json.dumps(q,indent=2),encoding='utf-8');print(json.dumps(q,indent=2))
