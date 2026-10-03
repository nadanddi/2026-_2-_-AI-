"""Synthetic scalar identity only, no model training or competition data."""
from pathlib import Path
import math,json
import numpy as np
OUT=Path(__file__).resolve().parent
y=np.array([1.,8.]);b=np.array([1.,2.]);r=y/b
ratio_mean=float(r.mean())
original_sse_mean=float(np.sum(b*y)/np.sum(b*b))
manual=math.fsum(float(u)*float(v) for u,v in zip(b,y))/math.fsum(float(u)**2 for u in b)
assert abs(original_sse_mean-manual)<1e-15
def original_loss(a):return math.fsum((float(yi)-float(bi)*a)**2 for yi,bi in zip(y,b))
def ratio_loss(a):return math.fsum((float(ri)-a)**2 for ri in r)
assert abs(ratio_mean-2.5)<1e-15
assert abs(original_sse_mean-3.4)<1e-15
assert abs(original_loss(ratio_mean)-11.25)<1e-12
assert abs(original_loss(original_sse_mean)-7.2)<1e-12
assert abs(ratio_loss(ratio_mean)-4.5)<1e-12
assert abs(ratio_loss(original_sse_mean)-6.12)<1e-12
assert abs(math.fsum(float(bi)*(float(bi)*original_sse_mean-float(yi)) for yi,bi in zip(y,b)))<1e-12
const=np.array([2.,2.]);rr=y/const
assert abs(float(np.sum(const*y)/np.sum(const*const))-float(rr.mean()))<1e-15
result=dict(status='PASS',scope='Synthetic arithmetic counterexample only',y=y.tolist(),b=b.tolist(),ratios=r.tolist(),ratio_sse_optimum=ratio_mean,original_sse_optimum=original_sse_mean,original_sse_at_ratio_optimum=original_loss(ratio_mean),original_sse_at_original_optimum=original_loss(original_sse_mean),ratio_sse_at_ratio_optimum=ratio_loss(ratio_mean),ratio_sse_at_original_optimum=ratio_loss(original_sse_mean),constant_b_optima_equal=True)
(OUT/'log_leaf_original_loss_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
