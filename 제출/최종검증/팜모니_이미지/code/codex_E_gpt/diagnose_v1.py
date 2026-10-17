import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent));import run_v2 as R
import numpy as np
from sklearn.model_selection import GroupKFold
F,rr,ds=R.assemble(R.POS+R.NEG);out=[]
for held in R.POS:
 _,hr=R.load(held);k=(ds!=held)&R.disjoint(rr,hr);y=np.array([int(r['label']) for r in rr]);grp=np.array([r['group'] for r in rr]);fr=[]
 for tr,te in GroupKFold(5).split(y[k],groups=grp[k]):fr.append(np.bincount(y[k][tr],minlength=2).tolist())
 out.append(dict(held=held,counts=np.bincount(y[k],minlength=2).tolist(),fold_train_counts=fr))
print(json.dumps(out),flush=True)
(Path(__file__).parent/'diagnosis.json').write_text(json.dumps(out,indent=2))
