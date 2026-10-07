"""Re-fit all nine CPU members and test raw outputs under future-input changes."""
import json,time
from blk_r3_baseline_v1 import model
from blk_baseline_data_v1 import *
from threadpoolctl import threadpool_limits
from checkpoint_v1 import atomic

def main():
    folder=HERE/'checkpoints/BLK_R3_v1'
    rp=folder/'registration.json';reg=json.loads(rp.read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==s for p,s in reg['dependencies'].items())
    assert all(sha(Path(env.DATA)/p)==s for p,s in reg['sources'].items())
    ctx=BLKContext(json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8')))
    _,tr,table=prepare_reference(ctx);ids=reg['ordered_query_ids'];q=prepare_query(ctx,ids,table)
    assert tr.row_id.tolist()==reg['ordered_train_ids']
    changes=[]
    for f in ['F13','F47']:
        fid=[r for r in ids if key(r)[0]==f]
        for fraction in [.25,.5,.75]:
            cut=key(fid[int((len(fid)-1)*fraction)])[1:]
            protected=[i for i,r in enumerate(ids) if key(r)[0]==f and key(r)[1:]<=cut]
            future=[r for r in ctx._query if key(r)[0]!=f or key(r)[1:]>cut]
            original={r:ctx._query[r] for r in future}
            try:
                for r in future:ctx._query[r]={c:(99999.0 if v is not None else None) for c,v in ctx._query[r].items()}
                alt=prepare_query(ctx,ids,table)
                pd.testing.assert_frame_equal(q.iloc[protected][FULL_R3],alt.iloc[protected][FULL_R3])
            finally:ctx._query.update(original)
            changes.append((f,cut,protected,alt))
    checks=[];started=time.monotonic()
    for seed in [47,1414,6464]:
        for name,cols in [('ET',FULL_R3),('LGB',BASE_R3),('MLP',BASE_R3)]:
            saved=json.loads((folder/f'{name}_seed{seed}.json').read_text(encoding='utf-8'))
            expected=np.array(saved['pred'],float)
            with threadpool_limits(limits=1):
                m=model(name,seed);m.fit(tr[cols],tr.sub_ec.to_numpy(float))
                if name=='ET':m.steps[-1][1].n_jobs=1
                pred=np.asarray(m.predict(q[cols]),float)
                reproduction=float(np.max(np.abs(pred-expected)));assert reproduction<=1e-6
                for f,cut,protected,alt in changes:
                    p=np.asarray(m.predict(alt[cols]),float)
                    diff=float(np.max(np.abs(p[protected]-pred[protected])));assert diff<=1e-6
                    checks.append({'seed':seed,'member':name,'farm':f,'cut':list(cut),'protected_rows':len(protected),'difference':diff})
                reversed_pred=np.asarray(m.predict(q.iloc[::-1][cols]),float)[::-1]
                assert np.max(np.abs(reversed_pred-pred))<=1e-6
            print(f'R3 future audit {name} seed{seed} PASS',flush=True)
    out=HERE/'BLK_R3_full_future_audit_v1.json';assert not out.exists()
    atomic(out,json.dumps({'status':'PASS','members':9,'future_other_farm_model_checks':len(checks),'checks':checks,
        'reproduced_original_raw_outputs':True,'reversed_query_order_checks':9,'registration_sha256':sha(rp),
        'code_sha256':sha(__file__),'heldout_truth_loaded':False,'duration_seconds':time.monotonic()-started},ensure_ascii=False))

if __name__=='__main__':main()
