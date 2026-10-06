from pathlib import Path
import sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import numpy as np
from leaf_core_v1 import leaf_weights,weight_shapley
t=np.array([[1,4],[1,5],[2,4]]);q=np.array([[1,4],[2,5]])
w=leaf_weights(t,q);np.testing.assert_allclose(w,[[.5,.25,.25],[0.,.5,.5]])
np.testing.assert_allclose(w@np.array([.2,.4,2.]),[.7,1.2]);np.testing.assert_allclose(w.sum(axis=1),1)
ix=[2,0,1];np.testing.assert_allclose(leaf_weights(t[ix],q)[:,np.argsort(ix)],w)
np.testing.assert_allclose(leaf_weights(t,q[::-1]),w[::-1])
try:leaf_weights(t,np.array([[99,4]]))
except ValueError:pass
else:raise AssertionError('Unrepresented leaf must fail')
coal=np.array([[1.,0.,0.],[.5,.5,0.],[.5,0.,.5],[0.,.5,.5]])
p=weight_shapley(coal,2);np.testing.assert_allclose(p,[[-.5,.5,0.],[-.5,0.,.5]])
np.testing.assert_allclose(p.sum(axis=0),coal[-1]-coal[0]);np.testing.assert_allclose(p.sum(axis=1),0)
print('PASS_LEAF_WEIGHT_AND_SHAPLEY_CONTRACT',flush=True)
