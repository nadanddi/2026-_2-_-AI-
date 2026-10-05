"""Numerical control: same MLP objective/features/train rows, higher solver accuracy."""
from pathlib import Path
import sys,json,math,importlib.util
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('investigation_precision',H/'run_v1.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D)
import mlp_audit_v1 as PA
np,pd=D.np,D.pd
def main():
    reg=D.load(H/'precision_registration_v1.json')
    for name,digest in reg['hashes'].items():assert D.sha(H/name)==digest
    assert reg['original_receipt_sha']==D.sha(H/'fit_receipt_v1.json');gradient_check=PA.toy();refs,sig,p=D.prepare();assert p==D.load(H/'preparation_v1.json');receipt=D.load(H/'fit_receipt_v1.json');verified=D.load(H/'verification_v2.json');assert verified['status'].startswith('PASS_DIAGNOSIS_AND_META270')
    dest=D.OUT/'MLP_precision';dest.mkdir(exist_ok=False);checks=[];frames=[];actual_fits=0
    for record in receipt['meta']:
        if record['mode']!='MLP':continue
        v,k,s,j=[record[n] for n in ['validator','fold','seed','meta_fold']];ref=refs[v,k];ti,vi=D.meta_split(ref,j);new=D.meta_ref(ref,s,ti,vi);original=D.load(D.OUT/f'meta_MLP_{v}_{k}_{s}_{j}.json')
        with D.M.threadpool_limits(limits=2):model,control=PA.refine(D,new,s,original)
        actual_fits+=int(control['optimized']);q,a,b,x,ok=D.B.pair(new,s,False);g0=np.where(ok,D.M.forward('MLP',x,original),0);g=np.where(ok,D.M.forward('MLP',x,model),0);p0=a+g0*(b-a);p1=a+g*(b-a);D.close(g,np.where(ok,D.M.forward('MLP',x[::-1],model)[::-1],0))
        y=q.sub_ec.to_numpy();control.update(validator=v,fold=k,seed=s,meta_fold=j,n=len(q),train_n=model['n'],max_probability_change=float(np.max(abs(g-g0))),max_prediction_change=float(np.max(abs(p1-p0))),old_sse=float(np.sum((p0-y)**2)),new_sse=float(np.sum((p1-y)**2)),base_sse=float(np.sum((a-y)**2)))
        scalar=math.fsum((float(aa)+float(gg)*(float(bb)-float(aa))-float(yy))**2 for aa,bb,gg,yy in zip(a,b,g,y));assert abs(scalar-control['new_sse'])<1e-10
        D.save(dest/f'{v}_{k}_{s}_{j}.json',dict(model=model,control=control,original_model_sha=record['model_sha']));checks.append(control)
        f=q[['row_id','farm','day','hour']].copy();f['seed']=s;f['fold']=k;f['meta_fold']=j;f['y']=y;f['A']=a;f['old']=p0;f['precise']=p1;frames.append(f);print('PRECISION',k,s,j,'COMPLETE',flush=True)
    assert len(checks)==90;allrows=pd.concat(frames,ignore_index=True);allrows.to_csv(dest/'predictions.csv',index=False);scores=[]
    for s,q in allrows.groupby('seed'):
        n=len(q);base=math.sqrt(math.fsum((float(aa)-float(yy))**2 for aa,yy in zip(q.A,q.y))/n);old=math.sqrt(math.fsum((float(aa)-float(yy))**2 for aa,yy in zip(q.old,q.y))/n);new=math.sqrt(math.fsum((float(aa)-float(yy))**2 for aa,yy in zip(q.precise,q.y))/n);scores.append(dict(seed=int(s),n=n,baseline_rmse=base,original_meta_rmse=old,precise_meta_rmse=new,original_change_pct=100*(old/base-1),precise_change_pct=100*(new/base-1),delta_rmse=new-old))
    D.save(H/'precision_v1.json',dict(status='PASS_PRECISION90_OBJECTIVE_GRADIENT',cells=90,actual_optimizer_fits=actual_fits,synthetic_gradient_error=gradient_check,max_old_gradient=max(r['old_gradient'] for r in checks),max_new_gradient=max(r['new_gradient'] for r in checks),max_prediction_change=max(r['max_prediction_change'] for r in checks),scores=scores,checks=checks,predictions_sha=D.sha(dest/'predictions.csv'),adoption=False));print('PRECISION90_PASS',flush=True)
if __name__=='__main__':main()
