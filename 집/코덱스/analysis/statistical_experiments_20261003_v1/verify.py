from support import *
import math,time

def main():
    while not (HERE/'result.json').exists():time.sleep(3)
    result=json.loads((HERE/'result.json').read_text(encoding='utf-8'));oof=pd.read_csv(OUT/'oof.csv');assert sha(OUT/'oof.csv')==result['oof_hash']
    for file,h in result['source_hashes'].items():assert sha(HERE/file)==h
    original_t=pd.read_csv(ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv');original_e=pd.read_csv(ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv')
    tlabels=original_t.drop_duplicates('row_id').set_index('row_id').sub_temp.to_dict();elabels=original_e.drop_duplicates('row_id').set_index('row_id').sub_ec.to_dict();checked=[]
    for score in result['scores']:
        d=oof[(oof.target==score['target'])&(oof.arm==score['arm'])&(oof.validator==score['validator'])&(oof.seed==score['seed'])&(oof.context==score['context'])];labels=tlabels if score['target']=='TEMP' else elabels
        assert all(abs(float(row.y)-float(labels[row.row_id]))<1e-12 for row in d.itertuples())
        a=math.sqrt(math.fsum((float(r.baseline)-float(r.y))**2 for r in d.itertuples())/len(d));b=math.sqrt(math.fsum((float(r.candidate)-float(r.y))**2 for r in d.itertuples())/len(d));assert abs(a-score['baseline_rmse'])<1e-12 and abs(b-score['candidate_rmse'])<1e-12
        checked.append(dict(arm=score['arm'],validator=score['validator'],seed=score['seed'],context=score['context'],delta_pct=100*(b/a-1)))
    split_checks=coef_checks=mix_checks=0;max_coef_gap=0.
    for cp in OUT.glob('*_cpu.npz'):
        z=dict(np.load(cp));it=set(z['inner_train_id']);cal=set(z['row_id']);ot=set(z['outer_train_id']);assert it<=ot and cal<=ot and not it&cal
        pairs=[(r[:3],int(r[4:7])) for r in it];qs={(r[:3],int(r[4:7])) for r in cal};assert all(f!=g or abs(d-e)>1 for f,d in pairs for g,e in qs);split_checks+=1
    for cp in OUT.glob('*_pfn_*.npz'):
        z=dict(np.load(cp));prefix=cp.name.split('_pfn_')[0];inp=dict(np.load(OUT/f'{prefix}_input.npz'));assert set(z['context_row_id'])<=set(inp['train_id']);assert not set(z['context_row_id'])&set(inp['query_id']);assert str(z['input_hash'])==sha(OUT/f'{prefix}_input.npz')
    for cp in OUT.glob('*_meta.npz'):
        z=dict(np.load(cp));ids=z['outer_row_id'];is_temp=cp.name.startswith('T_')
        prefix='_'.join(cp.name.split('_')[:3]);training=dict(np.load(OUT/f'{prefix}_cpu.npz'));assert not set(ids)&set(training['outer_train_id'])
        if is_temp:
            for xkey,coefkey,alpha in [('cal_z','bias_coef',100.),('cal_design','gate_coef',1000.)]:
                x=z[xkey];w=z['cal_w'];theta=np.linalg.solve(x.T@(w[:,None]*x)+alpha*np.eye(x.shape[1]),x.T@(w*z['cal_target']));gap=float(np.max(np.abs(theta-z[coefkey])));assert gap<1e-9;max_coef_gap=max(max_coef_gap,gap);coef_checks+=1
            p=z['outer_p'];ww=z['weights'];assert np.all(ww>=0) and np.max(np.abs(ww.sum(1)-1))<1e-12;assert np.max(np.abs(ww-z['old_weights'])-.1*z['outer_g'][:,None])<1e-12
            seed=int(cp.name.split('_')[-3]);context=cp.name.split('_')[-2];name=cp.name.split('_')[1];fold=int(cp.name.split('_')[2])
            for arm,pred in [('TGATE',np.array([math.fsum(float(a)*float(b) for a,b in zip(row,weights)) for row,weights in zip(p,ww)])),('TBIAS',(p*z['old_weights']).sum(1)+z['bias_delta'])]:
                d=oof[(oof.target=='TEMP')&(oof.arm==arm)&(oof.validator==name)&(oof.fold==fold)&(oof.seed==seed)&(oof.context==context)].set_index('row_id').reindex(ids);assert np.max(np.abs(pred-d.candidate.to_numpy()))<1e-11;mix_checks+=1
            cold=z['outer_g']==0;assert np.max(np.abs(ww[cold]-z['old_weights'][cold]),initial=0)<1e-12
        else:
            x=z['cal_design'];theta=np.linalg.solve(x.T@x+100*np.eye(x.shape[1]),x.T@z['cal_target']);gap=float(np.max(np.abs(theta-z['coef'])));assert gap<1e-9;max_coef_gap=max(max_coef_gap,gap);coef_checks+=1
            seed=int(cp.name.split('_')[-2]);name=cp.name.split('_')[1];fold=int(cp.name.split('_')[2]);d=oof[(oof.target=='EC')&(oof.validator==name)&(oof.fold==fold)&(oof.seed==seed)].set_index('row_id').reindex(ids)
            assert np.all((z['outer_source']==-1)|(z['outer_source']<d.day.to_numpy()));pred=np.clip(d.baseline.to_numpy()+np.clip(z['outer_design']@theta,-.15,.15),z['clip_lo'],z['clip_hi']);assert np.max(np.abs(pred-d.candidate.to_numpy()))<1e-11;mix_checks+=1
    assert len(checked)==39 and split_checks==34 and coef_checks==162 and mix_checks==162,(len(checked),split_checks,coef_checks,mix_checks)
    savej(HERE/'verification.json',dict(status='PASS',score_cells=checked,split_checks=split_checks,coef_checks=coef_checks,mix_checks=mix_checks,max_coef_gap=max_coef_gap,method='fresh labels/different fsum arithmetic and normal equations',source_hashes={p.name:sha(p) for p in HERE.glob('*.py')},oof_hash=sha(OUT/'oof.csv')))
    print('VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
