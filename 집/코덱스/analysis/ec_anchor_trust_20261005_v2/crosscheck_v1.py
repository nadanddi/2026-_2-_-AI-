"""Independent Decimal scores and manual farm-stratified bootstrap, no model import/fit."""
from pathlib import Path
import sys,csv,json,math,hashlib,argparse,collections
from decimal import Decimal,localcontext
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
def main(mode):
 p=ROOT/'집/코덱스/local/ec_anchor_trust_20261005_v2'/mode/'oof.csv';v=json.loads((H/f'verification_{mode}_v1.json').read_text(encoding='utf-8'));assert hashlib.sha256(p.read_bytes()).hexdigest()==v['aggregate'];groups=collections.defaultdict(list)
 with p.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):groups[r['validator'],int(r['seed'])].append(r)
 assert len(groups)==15;checks=[];boots={}
 with localcontext() as ctx:
  ctx.prec=45
  for score in v['scores']:
   rs=groups[score['validator'],score['seed']]
   for col in ['baseline','candidate']:
    z=float((sum((Decimal(r[col])-Decimal(r['y']))**2 for r in rs)/Decimal(len(rs))).sqrt());assert abs(z-score[col])<1e-12;checks.append(dict(validator=score['validator'],seed=score['seed'],column=col,decimal_rmse=z))
 for s in [7,101,2024]:
  rs=groups['DIAG10',s];daily=collections.defaultdict(list)
  for r in rs:daily[r['farm'],int(r['day'])].append((float(r['candidate'])-float(r['y']))**2-(float(r['baseline'])-float(r['y']))**2)
  rng=np.random.default_rng(20261003+s);tot=[0.]*20000;nn=[0]*20000
  for f in ['F13','F47']:
   days=sorted(d for ff,d in daily if ff==f);blocks=[[z for d in days[i:i+5] for z in daily[f,d]] for i in range(0,len(days),5)];ss=[math.fsum(b) for b in blocks];sizes=[len(b) for b in blocks];ix=rng.integers(len(blocks),size=(20000,len(blocks)))
   for j,draw in enumerate(ix):tot[j]+=math.fsum(ss[int(i)] for i in draw);nn[j]+=sum(sizes[int(i)] for i in draw)
  sample=np.array([a/n for a,n in zip(tot,nn)]);pw=sum(z>=0 for z in sample)/len(sample);ci=np.quantile(sample,[v['alpha'],1-v['alpha']]).tolist();assert pw==v['bootstrap'][str(s)]['p_worse'];assert max(abs(a-b) for a,b in zip(ci,v['bootstrap'][str(s)]['ci_adjusted']))<1e-12;boots[s]=dict(p_worse=pw,ci_adjusted=ci)
 target=H/f'crosscheck_{mode}_v1.json'
 with target.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_DECIMAL30_MANUAL_BOOT3',mode=mode,decimal_scores=checks,bootstrap=boots,fit=0,predict=0,source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
 print(mode,'INDEPENDENT_CROSSCHECK_PASS')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['GATE','RIDGE']);main(ap.parse_args().mode)
