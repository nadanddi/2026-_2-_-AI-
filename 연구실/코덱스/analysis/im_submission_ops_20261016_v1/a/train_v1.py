"""Fixed A2 token LR; first fold is the midpoint review checkpoint."""
import sys,json,warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('error',category=ConvergenceWarning)
HERE=Path(__file__).resolve().parent
mode=sys.argv[1]
with np.load(HERE/'samples_v1.npz') as z:
    X=z['X'].astype(np.float32);y=z['y'];own=z['own'];fold=z['fold']
assert np.isfinite(X).all() and set(y)=={0,1}
def fit(ix):
    assert set(y[ix])=={0,1}
    scaler=StandardScaler().fit(X[ix]);model=LogisticRegression(C=1,class_weight='balanced',max_iter=3000).fit(scaler.transform(X[ix]),y[ix])
    M=dict(mean=scaler.mean_.astype(np.float32),scale=scaler.scale_.astype(np.float32),coef=model.coef_.ravel().astype(np.float32),intercept=model.intercept_.astype(np.float32))
    assert all(np.isfinite(v).all() for v in M.values())
    return M
if mode=='first':
    ix=fold[own]!=0;M=fit(ix);np.savez(HERE/'fold0_model_v1.npz',**M)
    with (HERE/'first_fit_v1.json').open('x') as f:json.dump(dict(train_tokens=int(ix.sum()),positive=int(y[ix].sum()),negative=int((y[ix]==0).sum()),held_photos=int((fold==0).sum()),finite=True),f,indent=2)
    print('A2_FIRST_FIT_DONE',flush=True)
elif mode=='rest':
    assert (HERE/'first_fit_v1.json').is_file()
    for i in range(1,5):
        np.savez(HERE/f'fold{i}_model_v1.npz',**fit(fold[own]!=i));print('A2_FIT',i,flush=True)
    np.savez(HERE/'final_model_v1.npz',**fit(np.ones(len(y),bool)))
    print('A2_ALL_FIT_DONE',flush=True)
else:raise ValueError(mode)
