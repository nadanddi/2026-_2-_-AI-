from pathlib import Path
import sys,json,hashlib,warnings,os
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(H));import stage1_v2 as S
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from risk_core_v2 import design
L=S.L/'risk_inputs_v1';O=S.L/'risk_stage2_v1'
CONFIG=dict(label_threshold=.15,C=1.,tol=1e-8,max_iter=2000,threshold=.8,cap=.1,vent_zero=.8,fan_mean=10.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as stream:json.dump(v,stream,ensure_ascii=False,indent=2,allow_nan=False)
def prepare():
    assert not (S.O/'worker.lock').exists()
    assert json.loads((H/'stage1_receipt_v2.json').read_text())['status']=='COMPLETE_DIAGNOSTIC'
    audits=sorted(H.glob('verify_stage1_complete_*.json'));assert audits and json.loads(audits[-1].read_text(encoding='utf-8'))['status']=='PASS_COMPLETE_DIAGNOSTIC'
    inputs=json.loads((H/'risk_input_preparation_v1.json').read_text());assert len(inputs['files'])==12 and not inputs['full80']
    files=[];names=None
    for rec in inputs['files']:
        k,seed=rec['k'],rec['seed'];tp=L/f'train_{k}_{seed}.csv';qp=L/f'query_{k}_{seed}.csv'
        assert sha(tp)==rec['train_sha'] and sha(qp)==rec['query_sha']
        tr=pd.read_csv(tp,float_precision='round_trip');q=pd.read_csv(qp,float_precision='round_trip');assert not set(tr.row_id)&set(q.row_id)
        for frame in [tr,q]:assert frame.groupby(['farm','day']).size().eq(24).all()
        x=design(tr,S.M.FULL);z=design(q,S.M.FULL);assert list(x)==list(z) and len(x.columns)==50
        if names is None:names=list(x)
        assert names==list(x)
        files.append(rec)
    result=dict(status='PREPARED_PARTIAL_RISK_ONLY',config=CONFIG,base_whitelist=S.M.FULL,feature_columns=names,inputs=files,source_sha=sha(__file__),helper_sha=sha(H/'risk_core_v2.py'),protocol_sha=sha(H/'PROTOCOL2_v1.md'),fit=0,full80=False)
    dest=H/'stage2_preparation_v1.json'
    if dest.exists():assert json.loads(dest.read_text(encoding='utf-8'))==result
    else:write(dest,result)
    return result
def main():
    prep=prepare()
    if '--prepare' in sys.argv:print('STAGE2_PREPARED_FIT0',flush=True);return
    O.mkdir(exist_ok=False);write(O/'worker.lock',dict(pid=os.getpid(),source_sha=sha(__file__)))
    outputs=[];records=[]
    for rec in prep['inputs']:
        k,seed=rec['k'],rec['seed'];tr=pd.read_csv(L/f'train_{k}_{seed}.csv',float_precision='round_trip');q=pd.read_csv(L/f'query_{k}_{seed}.csv',float_precision='round_trip')
        x=design(tr,prep['base_whitelist']);z=design(q,prep['base_whitelist']);assert list(x)==prep['feature_columns']
        y=(tr.A-tr.sub_ec>CONFIG['label_threshold']).astype(int).to_numpy()
        weights=1/tr.groupby(['farm','day']).A.transform('size').to_numpy(float)
        meta=dict(k=k,seed=seed,train_sha=rec['train_sha'],query_sha=rec['query_sha'],train_rows=len(tr),query_rows=len(q),positives=int(y.sum()),sum_weight=float(weights.sum()),columns=list(x),constant=len(np.unique(y))<2)
        if meta['constant']:risk=np.full(len(q),float(y.mean()));meta['constant_score']=float(y.mean())
        else:
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=CONFIG['C'],solver='lbfgs',tol=CONFIG['tol'],max_iter=CONFIG['max_iter'],random_state=seed))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                with threadpool_limits(limits=2):model.fit(x,y,logisticregression__sample_weight=weights);risk=model.predict_proba(z)[:,1]
            im,sc,lr=[v for _,v in model.steps];assert int(lr.n_iter_[0])<CONFIG['max_iter']
            assert np.max(abs(risk[:8]-model.predict_proba(z.iloc[:8])[:,1]))<1e-12
            meta.update(median=im.statistics_.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),iterations=int(lr.n_iter_[0]),classes=lr.classes_.tolist())
        assert np.isfinite(risk).all() and np.all((risk>=0)&(risk<=1))
        write(O/f'fit_{k}_{seed}.json',meta)
        out=q[['row_id','farm','day','hour','sub_ec','A','smooth_lgb','act_vent_tdz','act_circfan_tdm']].copy();out['risk']=risk;out['k']=k;out['seed']=seed;outputs.append(out);records.append(meta)
        print('RISK_FIT_COMPLETE',k,seed,flush=True)
    dest=O/'risk_rows.csv';pd.concat(outputs,ignore_index=True).to_csv(dest,index=False)
    write(H/'stage2_receipt_v1.json',dict(status='COMPLETE_PARTIAL_RISK_FIT',cells=len(records),fit=sum(not r['constant'] for r in records),rows=sum(r['query_rows'] for r in records),output_sha=sha(dest),source_sha=sha(__file__),prep_sha=sha(H/'stage2_preparation_v1.json'),full80=False,correction=False,adoption=False))
    (O/'worker.lock').unlink();print('STAGE2_COMPLETE',flush=True)
if __name__=='__main__':main()
