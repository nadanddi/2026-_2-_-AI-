from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import runner_v4 as R
import pandas as pd,numpy as np
def read(p):return pd.read_csv(p,float_precision='round_trip',dtype={'seed':str})
def main():
    stage='ablation' if '--ablation' in sys.argv else 'baseline'
    receipt=H/('ablation_receipt_v1.json' if stage=='ablation' else 'baseline_receipt_v4.json');rr=json.loads(receipt.read_text(encoding='utf-8'));assert rr['rows_sha']==R.sha(R.L/(stage+'_rows.csv'))
    raw,jobs,_=R.prepare();x=raw[['row_id','act_vent','act_circfan']].copy();x['farm']=x.row_id.str[:3];x['day']=x.row_id.str[4:7].astype(int)
    state=x.groupby(['farm','day']).agg(ventzero=('act_vent',lambda s:float((s==0).mean())),fanmean=('act_circfan','mean')).reset_index()
    base=read(R.L/'baseline_rows.csv');current=[];older=[]
    for k,(t,q,pfn) in jobs.items():
        b=[];old=[]
        for s in R.SEEDS:
            with np.load(R.L/'base'/f'{k}_{s}.npz',allow_pickle=False) as c:
                assert c['row_id'].tolist()==q.row_id.tolist();b.append(c['et'].copy())
            with np.load(R.C/f'DIAG10_{k}_r3_{s}.npz',allow_pickle=False) as c:
                assert c['row_id'].tolist()==q.row_id.tolist();old.append(.48*c['raw_et']+.24*c['raw_lgb']+.08*c['raw_mlp']+.2*pfn)
        for i,s in enumerate([*R.SEEDS,'ensemble']):
            f=base[(base.k==k)&(base.seed==str(s))].copy();f=f.set_index('row_id').loc[q.row_id].reset_index();f['raw_et']=b[i] if i<3 else np.mean(b,axis=0);current.append(f)
            a=q[['row_id','farm','day','hour','sub_ec']].copy();a['k']=k;a['arm']='OLD_SEASON';a['seed']=str(s);mix=old[i] if i<3 else np.mean(old,axis=0);a['prediction']=np.clip(R.M.shrink(mix,q),t.sub_ec.min(),t.sub_ec.max());older.append(a)
    base=pd.concat(current,ignore_index=True);parts=[base,pd.concat(older,ignore_index=True)]
    if stage=='ablation':parts.append(read(R.L/'ablation_rows.csv'))
    rows=pd.concat(parts,ignore_index=True);assert rows[['arm','seed','row_id']].duplicated().sum()==0
    rows['error']=rows.prediction-rows.sub_ec;rows['squared_error']=rows.error**2
    keys=['arm','seed','farm','day'];g=rows.groupby(keys)
    days=g.agg(n=('row_id','size'),truth=('sub_ec','mean'),prediction=('prediction','mean'),bias=('error','mean'),sse=('squared_error','sum'),raw_et=('raw_et','mean'),k=('k','first')).reset_index();assert (days.n==24).all()
    days['rmse']=np.sqrt(days.sse/days.n);days['sse_day']=days.n*days.bias**2;days['sse_shape']=days.sse-days.sse_day
    days=days.merge(state,on=['farm','day'],validate='many_to_one');days['ordinary']=days.truth<1;days['closed']=days.ventzero>=.8;days['fanlow']=days.fanmean<10;days['pass2']=days.day>=179
    selected=days.farm.eq('F47')&days.day.eq(161)
    masks={'all':np.ones(len(days),bool),'ordinary':days.ordinary,'high':~days.ordinary,'ordinary_closed61':days.ordinary&days.closed,'ordinary_other268':days.ordinary&~days.closed,'ordinary_closed_fanlow50':days.ordinary&days.closed&days.fanlow,'pass1':~days.pass2,'pass2_public46':days.pass2,'F13':days.farm.eq('F13'),'F47':days.farm.eq('F47'),'excluding_F47_161':~selected,'ordinary_excluding_F47_161':days.ordinary&~selected}
    stats=[]
    for group,mask in masks.items():
        for (arm,seed),d in days[mask].groupby(['arm','seed']):
            stats.append(dict(group=group,arm=arm,seed=seed,days=len(d),rows=int(d.n.sum()),sse=float(d.sse.sum()),rmse=float(np.sqrt(d.sse.sum()/d.n.sum())),bias=float(d.bias.mean()),good_days=int((d.rmse<=.1).sum()),severe_days=int((d.rmse>=.2).sum()),day_level_sse_fraction=float(d.sse_day.sum()/d.sse.sum())))
    stats=pd.DataFrame(stats);ref=stats[stats.arm=='BASE'][['group','seed','sse','rmse']].rename(columns={'sse':'base_sse','rmse':'base_rmse'});stats=stats.merge(ref,on=['group','seed'],validate='many_to_one');stats['delta_sse']=stats.sse-stats.base_sse;stats['rmse_percent']=100*(stats.rmse/stats.base_rmse-1)
    case=days[[(f,int(d)) in [('F47',160),('F47',161),('F13',98),('F13',112)] for f,d in zip(days.farm,days.day)]].copy()
    interactions=[]
    mean=base[base.seed!='ensemble'].groupby('row_id').prediction.mean();actual=base[base.seed=='ensemble'].set_index('row_id').prediction;diff=actual-mean
    interactions.append(dict(kind='actual_ensemble_vs_mean_post',arm='BASE',changed_rows=int((abs(diff)>1e-12).sum()),maxdiff=float(abs(diff).max()),mean_absdiff=float(abs(diff).mean())))
    for arm in (ARMS if stage=='ablation' else []):
        a=rows[rows.arm==arm];mean=a[a.seed!='ensemble'].groupby('row_id').prediction.mean();actual=a[a.seed=='ensemble'].set_index('row_id').prediction;diff=actual-mean
        interactions.append(dict(kind='actual_ensemble_vs_mean_post',arm=arm,changed_rows=int((abs(diff)>1e-12).sum()),maxdiff=float(abs(diff).max()),mean_absdiff=float(abs(diff).mean())))
        for seed in [*map(str,R.SEEDS),'ensemble']:
            b=base[base.seed==seed].set_index('row_id');v=a[a.seed==seed].set_index('row_id').loc[b.index]
            assert np.array_equal(b.sub_ec,v.sub_ec)
            interactions.append(dict(kind='ET_change_and_SG2_interaction',arm=arm,seed=seed,changed_et_rows=int((abs(v.raw_et-b.raw_et)>1e-12).sum()),changed_final_rows=int((abs(v.prediction-b.prediction)>1e-12).sum()),gate_flips=int((v.gate!=b.gate).sum()),reference_candidate_changed=int(((v.candidate_day.fillna(-1)!=b.candidate_day.fillna(-1))|(v.candidate_ec.fillna(-1)!=b.candidate_ec.fillna(-1))).sum()),max_direct_pre_sg2_change=float(abs(v.pre_sg2-b.pre_sg2).max()),max_sg2_interaction=float(abs((v.sg2_raw-v.pre_sg2)-(b.sg2_raw-b.pre_sg2)).max())))
    D=H/(stage+'_score_v1');D.mkdir(exist_ok=False)
    for name,frame in [('days',days),('groups',stats),('cases',case),('interactions',pd.DataFrame(interactions))]:frame.to_csv(D/(name+'.csv'),index=False)
    tags={}
    if stage=='ablation':
        c=case[(case.farm=='F47')&(case.day==161)&case.seed.ne('ensemble')];basebias=(c[c.arm=='BASE'].set_index('seed').raw_et-c[c.arm=='BASE'].set_index('seed').truth)
        for arm in ARMS:
            ab=c[c.arm==arm].set_index('seed');bias=ab.raw_et-ab.truth;tags[arm]=dict(SELECTED_CASE_DEPENDENCE=bool((abs(bias)<abs(basebias)).all()),RESIDUAL_STATE_CONFUSION=bool((bias>.2).all()),raw_et_day_bias=bias.to_dict())
    R.write(D/'completion.json',dict(status='SCORED_'+stage.upper(),input_sha=R.sha(R.L/(stage+'_rows.csv')),source_sha=R.sha(__file__),tags=tags,files={p.name:R.sha(p) for p in D.glob('*.csv')},adoption=False));print('SCORE_COMPLETE',stage,flush=True)
ARMS=['D1','D2']
if __name__=='__main__':main()
