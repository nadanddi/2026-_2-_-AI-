"""Pure arithmetic examples only. No model fit or competition labels."""
from pathlib import Path
import json,math
import numpy as np
OUT=Path(__file__).resolve().parent
examples=[]
for name,r,leaf in [('pure_leaves',[1.,4.],[0,1]),('duplicate_features',[1.,4.],[0,0]),('skew_not_median',[1.,1.,4.],[0,0,0])]:
    r=np.array(r);leaf=np.array(leaf)
    cnt=np.bincount(leaf);sums=np.bincount(leaf,weights=r)
    leaf_arithmetic=sums/cnt
    leaf_log=np.bincount(leaf,weights=np.log(r))/cnt
    for k in np.unique(leaf):
        rows=np.flatnonzero(leaf==k)
        manual=math.fsum(float(r[i]) for i in rows)/len(rows)
        logmanual=math.fsum(math.log(float(r[i])) for i in rows)/len(rows)
        assert abs(leaf_arithmetic[k]-manual)<1e-14
        assert abs(leaf_log[k]-logmanual)<1e-14
        assert leaf_arithmetic[k]+1e-14>=math.exp(leaf_log[k])
    # Equal weighting across these leaves is an illustrative forest partition.
    original=math.exp(float(leaf_log.mean()))
    inverse_each=float(np.exp(leaf_log).mean())
    proposed=float(leaf_arithmetic.mean())
    assert proposed+1e-14>=inverse_each>=original-1e-14
    examples.append(dict(name=name,ratios=r.tolist(),assignment=leaf.tolist(),original_exp_meanlog=original,inverse_each_tree=inverse_each,proposed_mean_leaf_ratio=proposed,ratio_median=float(np.median(r))))
assert abs(examples[0]['proposed_mean_leaf_ratio']-2.5)<1e-14
assert abs(examples[1]['inverse_each_tree']-2.)<1e-14
assert abs(examples[1]['proposed_mean_leaf_ratio']-2.5)<1e-14
assert abs(examples[2]['original_exp_meanlog']-4**(1/3))<1e-14
assert examples[2]['ratio_median']==1.
result=dict(status='PASS',scope='Synthetic arithmetic only, no fitting or real-data performance inference',examples=examples,proof_checks=['leaf sums vs scalar fsum','leaf Jensen','forest Jensen','duplicate-leaf counterexample','exp(mean log) not necessarily median'])
(OUT/'log_leaf_math_check_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
