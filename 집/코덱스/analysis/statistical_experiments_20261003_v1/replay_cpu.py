from support import *
from sklearn.neural_network import MLPRegressor
def main():
    lab,ct,phc,w,folds,_=loadtemp();name,k,fd=folds[0];tm,_=common.split_mask(lab,fd);tr=lab[tm];im,iv=inner(tr);a,b=tr[im],tr[iv];iw=w[tm][im];cold_v5.SEED=7
    with threadpool_limits(limits=2):
        mem=temp_members(a,b,ct,phc,iw);base=.65*mem['res']+.25*mem['ridge']+.1*mem['nys'];codex=TM.codex_fit_predict(a,b,iw,726)
    old=dict(np.load(OUT/'T_DIAG10_0_cpu.npz'));assert np.array_equal(b.row_id,old['row_id']);tb=float(np.max(np.abs(base-old['base_7'])));tc=float(np.max(np.abs(codex-old['codex_7'])));assert max(tb,tc)<1e-9
    lab,core,wv,folds,_=loadec();name,k,tm,_=folds[0];tr=lab[tm];im,iv=inner(tr);a,b=seasonal(tr[im],tr[iv],wv);full=['season' if c=='day' else c for c in core.FULL];bc=['season' if c=='day' else c for c in core.BASE]
    models=[(core.et(7),full),(core.lg(7,'tweedie'),bc),(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=7)),bc)];pred=[]
    for model,cols in models:
        with threadpool_limits(limits=2):model.fit(a[cols],a.sub_ec.to_numpy());pred.append(model.predict(b[cols]))
    r=.6*pred[0]+.3*pred[1]+.1*pred[2];old=dict(np.load(OUT/'E_DIAG10_0_cpu.npz'));assert np.array_equal(b.row_id,old['row_id']);eg=float(np.max(np.abs(r-old['r3_7'])));assert eg<1e-9
    result=dict(status='PASS',temp_base_first_fold_maxdiff=tb,temp_codex_first_fold_maxdiff=tc,ec_r3_first_fold_maxdiff=eg,source_hash=sha(__file__));savej(HERE/'cpu_replay.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
