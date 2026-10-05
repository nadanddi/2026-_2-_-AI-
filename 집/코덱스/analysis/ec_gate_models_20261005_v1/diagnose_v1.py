"""Post-hoc error analysis only; no model selection or new prediction rules."""
from pathlib import Path
import sys,csv,math,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
OUT=ROOT/'집/코덱스/local'/H.name
def main():
    completed=json.loads((H/'verification_all_v1.json').read_text(encoding='utf-8'));assert completed['status']=='PASS_ALL198';result=[]
    for mode in ['LR','LGB','MLP']:
        v=json.loads((H/f'verification_{mode}_v1.json').read_text(encoding='utf-8'));path=OUT/mode/'oof.csv';assert hashlib.sha256(path.read_bytes()).hexdigest()==v['aggregate_sha']
        with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        for s in [7,101,2024]:
            d=[r for r in rows if r['validator']=='DIAG10' and int(r['seed'])==s];a=np.array([float(r['A']) for r in d]);b=np.array([float(r['B']) for r in d]);y=np.array([float(r['y']) for r in d]);g=np.array([float(r['g']) for r in d]);p=np.array([float(r['candidate']) for r in d]);cost=(a-y)**2-(b-y)**2;delta=b-a
            oracle=np.divide(y-a,delta,out=np.zeros(len(d)),where=delta!=0);oracle=np.clip(oracle,0,1);op=a+oracle*delta
            # Separate scalar implementation of the per-row squared-loss minimizer.
            manual=[min(1.,max(0.,(float(yy)-float(aa))/(float(bb)-float(aa)))) if float(bb)!=float(aa) else 0. for aa,bb,yy in zip(a,b,y)];assert np.max(abs(oracle-manual))<1e-12
            orrm=math.sqrt(math.fsum((float(aa)+float(gg)*(float(bb)-float(aa))-float(yy))**2 for aa,bb,gg,yy in zip(a,b,manual,y))/len(y));assert abs(orrm-np.sqrt(np.mean((op-y)**2)))<1e-12
            daily={}
            for r,aa,bb,yy,pp,gg in zip(d,a,b,y,p,g):
                key=(r['farm'],int(r['day']));rec=daily.setdefault(key,dict(y=[],a=[],b=[],p=[],g=[]));rec['y'].append(float(yy));rec['a'].append(float(aa));rec['b'].append(float(bb));rec['p'].append(float(pp));rec['g'].append(float(gg))
            top=[]
            for (f,day),rec in daily.items():
                change=math.fsum((pp-yy)**2-(aa-yy)**2 for aa,pp,yy in zip(rec['a'],rec['p'],rec['y']));top.append(dict(farm=f,day=day,loss_change=change,mean_y=math.fsum(rec['y'])/len(rec['y']),mean_gate=math.fsum(rec['g'])/len(rec['g'])))
            positive=cost>0;negative=cost<0
            record=dict(mode=mode,seed=s,n=len(d),days=len(daily),B_better_rows=int(positive.sum()),A_better_rows=int(negative.sum()),gate_mean_when_B_better=float(g[positive].mean()),gate_mean_when_A_better=float(g[negative].mean()),weighted_gate_B_better=float(np.average(g[positive],weights=cost[positive])),weighted_gate_A_better=float(np.average(g[negative],weights=-cost[negative])),actual_improved_rows=int((((a-y)**2-(p-y)**2)>0).sum()),baseline_rmse=float(np.sqrt(np.mean((a-y)**2))),candidate_rmse=float(np.sqrt(np.mean((p-y)**2))),oracle_rmse=orrm,oracle_change_pct=100*(orrm/np.sqrt(np.mean((a-y)**2))-1),top_loss_days=sorted(top,key=lambda r:r['loss_change'],reverse=True)[:10])
            assert abs(math.fsum(t['loss_change'] for t in top)-math.fsum((float(pp)-float(yy))**2-(float(aa)-float(yy))**2 for aa,pp,yy in zip(a,p,y)))<1e-10;result.append(record)
    payload=dict(status='PASS_POSTHOC_SCALAR_ORACLE_AND_DAILY_SUM',label='oracle uses outer y and is never an inference rule or adoption candidate',rows=result)
    with (H/'diagnostic_v1.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    print('DIAGNOSTIC9_PASS',flush=True)
if __name__=='__main__':main()
