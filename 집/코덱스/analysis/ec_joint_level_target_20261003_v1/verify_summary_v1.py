from pathlib import Path
import csv,json,math
H=Path(__file__).resolve().parent
with open(H/'target_loss_check_v1.csv',newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert len(rows)==22 and len({(r['validator'],r['fold']) for r in rows})==22
def median(values):
 s=sorted(values);n=len(s);return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
a=median([float(r['day_mean_var'])/float(r['current_var']) for r in rows])
b=median([float(r['day_mean_var'])/float(r['joint_root_loss']) for r in rows])
old=json.loads((H/'target_loss_verification_v1.json').read_text())
assert abs(a-old['median_day_fraction'])<1e-12 and abs(b-old['median_joint_day_fraction'])<1e-12
for r in rows:
 assert abs(float(r['current_var'])-float(r['day_mean_var'])-float(r['within_var']))<1e-12
 assert abs(float(r['joint_root_loss'])-float(r['day_mean_var'])-.5*float(r['within_var']))<1e-12
z=dict(status='PASS',independent_csv_checks=46,folds=22,manual_median_day_fraction=a,manual_median_joint_day_fraction=b,scope='scalar CSV ratios and sorted median vs NumPy/Pandas summary; training-root label variance only')
(H/'summary_verification_v1.json').write_text(json.dumps(z,indent=2),encoding='utf-8');print(json.dumps(z,indent=2))
