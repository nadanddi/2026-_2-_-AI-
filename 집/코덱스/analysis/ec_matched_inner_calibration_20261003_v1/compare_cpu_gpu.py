from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
O=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1';c=dict(np.load(O/'DIAG10_0_pfn_1.npz'));g=dict(np.load(O/'gpu_equivalence_first.npz'))
assert np.array_equal(c['row_id'],g['row_id']) and np.array_equal(c['context_row_id'],g['context_row_id'])
d=c['prediction']-g['prediction'];n=len(d);rm=math.sqrt(math.fsum(float(x)**2 for x in d)/n);assert abs(rm-float(np.sqrt(np.mean(d*d))))<1e-12
out=dict(status='PASS',rows=n,maxdiff=float(np.max(np.abs(d))),rmsdiff=rm,threshold=1e-5,equivalence_first_context_only=bool(np.max(np.abs(d))<1e-5),limitations=['first fold/context only, not all GPU outputs certified','candidate gain requires full public + future confirmation','CPU baseline retained'])
(H/'cpu_gpu_comparison_v1.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
