"""Independent objective/scaler validation, no fitting, after complete mode."""
import sys,math,json,importlib.util,argparse
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('trust_learning_runner',H/'run_v6.py');R=importlib.util.module_from_spec(sp);sp.loader.exec_module(R)
np=R.np
def main(mode):
 refs,prep=R.preparation();assert R.load(H/'preparation_v6.json')==prep;receipt=R.load(H/f'fit_{mode}_v1.json');assert len(receipt['manifest'])==66;checks=[]
 for rec in receipt['manifest']:
  v,k,s=rec['validator'],rec['fold'],rec['seed'];ref=refs[v,k];b=ref['b'];x,ok,delta=R.design(b,ref['inner'][s],ref['NB'],ref['QB']);base=ref['inner'][s];y=b.sub_ec.to_numpy();trial=np.clip(base+delta,*ref['innerbounds']);cost=(base-y)**2-(trial-y)**2;sel=ok&(abs(cost)>1e-14) if mode=='GATE' else ok
  xx=x[sel];means=np.array([math.fsum(xx[:,j])/len(xx) for j in range(xx.shape[1])]);scales=np.array([math.sqrt(math.fsum((float(t)-means[j])**2 for t in xx[:,j])/len(xx)) for j in range(xx.shape[1])]);scales[scales==0]=1
  m=R.load(R.OUT/mode/f'{v}_{k}_{s}_fit.json');assert m['eligible_ids']==R.ids(b.loc[sel,'row_id']) and m['features_sha']==R.ar(x[sel]) and m['cost_sha']==R.ar(cost[sel]) and m['mode']==mode and m['seed']==s;R.close(m['mean'],means);R.close(m['scale'],scales);assert m['n']==int(sel.sum()) and m['positive']==int((cost[sel]>0).sum());a=(xx-means)/scales
  if m['model_none']:assert mode=='GATE' and m['positive'] in [0,m['n']];grad=0.
  else:
   coef=np.asarray(m['coef']).reshape(-1);inter=float(np.asarray(m['intercept']).reshape(-1)[0]);lin=a@coef+inter
   if mode=='GATE':
    label=(cost[sel]>0).astype(float);weights=abs(cost[sel]);weights=weights/(math.fsum(weights)/len(weights));p=1/(1+np.exp(-lin));res=weights*(p-label);g=a.T@res+coef;gi=math.fsum(res);grad=float(max(np.max(abs(g)),abs(gi)));assert grad<.001,(v,k,s,grad)
   else:
    res=lin-(y-base)[sel];g=a.T@res+100*coef;gi=math.fsum(res);grad=float(max(np.max(abs(g)),abs(gi)));assert grad<1e-7,(v,k,s,grad)
  checks.append(dict(validator=v,fold=k,seed=s,eligible_ids=R.ids(b.loc[sel,'row_id']),target_cost_sha=R.ar(cost[sel]),feature_sha=R.ar(x[sel]),objective_gradient_max=grad))
 R.save(H/f'learning_check_{mode}_v1.json',dict(status='PASS_SCALER_OBJECTIVE_NOFIT',mode=mode,cells=len(checks),checks=checks,verifier=R.sha(Path(__file__)),fit=0,predict=0,source=R.sha(H/'run_v6.py')));print(mode,'LEARNING_CHECK_PASS',max(c['objective_gradient_max'] for c in checks))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['GATE','RIDGE'],required=True);main(ap.parse_args().mode)

