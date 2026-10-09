"""Final training-data deletion rule and target integrity; no holdout scoring."""
import sys,json,csv,math,hashlib
from pathlib import Path
from collections import defaultdict
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'집/코덱스/local'/H.name
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
D=L/'dataset_full_train';rows=list(csv.DictReader((Path(env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='')))
y={r['row_id']:float(r['sub_ec']) for r in rows if r['row_id'].startswith(('F13_','F47_'))};assert len(y)==9600
parts=defaultdict(list)
for rid,v in y.items():parts[rid.rsplit('_',1)[0]].append(v)
assert len(parts)==400 and all(len(v)==24 for v in parts.values())
with np.load(D/'selector_snapshot_v1.npz',allow_pickle=False) as s:keys=s['train_keys'].tolist();z=s['train_z'].copy();targets=s['train_y'].copy()
assert len(keys)==400 and np.isfinite(z).all()
gap=max(abs(targets[i]-math.fsum(parts[key])/24) for i,key in enumerate(keys));assert gap<1e-12
fk=[key.split('_') for key in keys];selected=set()
for i,(farm,day) in enumerate(fk):
    distances=np.sqrt(np.mean((z-z[i])**2,axis=1));eligible=[j for j,(ff,dd) in enumerate(fk) if ff==farm and abs(int(dd)-int(day))>1]
    near=sorted(eligible,key=lambda j:(distances[j],fk[j][0],int(fk[j][1])))[:5]
    neighbor=float(np.median(targets[near]))
    if targets[i]>=1 and targets[i]-neighbor>.5:selected.add((farm,int(day)))
removed=list(csv.DictReader((D/'removed_days_v1.csv').open(encoding='utf8',newline='')));assert selected=={(r['farm'],int(r['day'])) for r in removed}
cy=list(csv.DictReader((D/'train_y_clean_v1.csv').open(encoding='utf8',newline='')));cx=list(csv.DictReader((D/'train_X_clean_v1.csv').open(encoding='utf8',newline='')))
assert len(cx)==len(cy)==9360 and [r['row_id'] for r in cx]==[r['row_id'] for r in cy]
retained={rid for rid in y if (rid[:3],int(rid[4:7])) not in selected};assert set(r['row_id'] for r in cy)==retained
pointgap=max(abs(float(r['sub_ec'])-y[r['row_id']]) for r in cy);assert pointgap<1e-12
out=dict(status='PASS',original_rows=9600,original_days=400,removed_days=len(selected),removed_rows=9600-len(cx),clean_rows=len(cx),clean_days=len(cx)//24,daily_target_maxdiff=gap,retained_target_maxdiff=pointgap,full_selector_snapshot_NN_rule_replayed=True,full_standardized_feature_matrix_source_reviewed_not_independently_rebuilt=True,consumed40_prediction_rescoring=False,script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (H/'full_cleaning_recheck_v1.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
