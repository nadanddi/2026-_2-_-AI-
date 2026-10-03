"""Fixed trained-forest mechanism audit; no tuning, fitting, or candidate adoption."""
from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd

def main():
    dest=H/'mechanism_audit_v1.json';assert not dest.exists()
    paths=list((ROOT/'집/코덱스/local/ec_log_partition_mean_20261004_v1').glob('*.csv'));assert len(paths)==66
    o=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True)
    lab,core,wv,folds,outer=S.loadec();bounds={(v,k):(float(lab[tm].sub_ec.min()),float(lab[tm].sub_ec.max())) for v,k,tm,vm in folds}
    assert len(o)==83160 and (o.new_et_raw>=o.old_log_raw-1e-10).all()
    parts=[];clip_counts={'arithmetic_component_outside':0,'geometric_component_outside':0,'arithmetic_final_clipped':0,'geometric_final_clipped':0}
    for (v,k,s),g in o.groupby(['validator','fold','seed'],sort=False):
        g=g.copy().reset_index(drop=True);lo,hi=bounds[(v,k)]
        ar=core.shrink(g.new_et_raw.to_numpy(),g);ge=core.shrink(g.old_log_raw.to_numpy(),g)
        apre=g.baseline.to_numpy()+.48*(ar-g.old_et_shrunk.to_numpy())
        gpre=g.baseline.to_numpy()+.48*(ge-g.old_et_shrunk.to_numpy())
        assert np.max(abs(np.clip(apre,lo,hi)-g.candidate.to_numpy()))<1e-12
        g['geometric_same_recipe']=np.clip(gpre,lo,hi)
        assert (g.candidate>=g.geometric_same_recipe-1e-12).all()
        clip_counts['arithmetic_component_outside']+=int(((ar<lo)|(ar>hi)).sum())
        clip_counts['geometric_component_outside']+=int(((ge<lo)|(ge>hi)).sum())
        clip_counts['arithmetic_final_clipped']+=int(((apre<lo)|(apre>hi)).sum())
        clip_counts['geometric_final_clipped']+=int(((gpre<lo)|(gpre>hi)).sum())
        parts.append(g)
    o=pd.concat(parts,ignore_index=True);cells=[]
    stored=pd.read_csv(H/'scores_v1.csv').set_index(['validator','seed'])
    for (v,s),g in o.groupby(['validator','seed']):
        rm={c:math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g[c]))/len(g)) for c in ['baseline','geometric_same_recipe','candidate']}
        assert abs(rm['baseline']-stored.loc[(v,s),'baseline'])<1e-12 and abs(rm['candidate']-stored.loc[(v,s),'candidate'])<1e-12
        delta=g.candidate.to_numpy()-g.baseline.to_numpy();err=g.baseline.to_numpy()-g.y.to_numpy()
        slope=2*math.fsum(float(e)*float(d) for e,d in zip(err,delta))/len(g)
        sq=math.fsum(float(d)**2 for d in delta)/len(g)
        assert abs(slope+sq-(rm['candidate']**2-rm['baseline']**2))<1e-12
        uplift=g.candidate.to_numpy()-g.geometric_same_recipe.to_numpy()
        cells.append(dict(validator=v,seed=int(s),n=len(g),rmse=rm,
                          arithmetic_vs_geometric_change_pct=100*(rm['candidate']/rm['geometric_same_recipe']-1),
                          direction_derivative_at_zero=slope,direction_quadratic_term=sq,
                          mean_final_jensen_uplift=float(np.mean(uplift)),max_final_jensen_uplift=float(np.max(uplift))))
    day=o[o.validator=='DIAG10'].groupby(['seed','farm','day']).agg(y=('y','mean'),baseline=('baseline','mean'),candidate=('candidate','mean'),geometric=('geometric_same_recipe','mean')).reset_index()
    rows=[]
    for s in [7,101,2024]:
        for seg,m in [('high',day.y>=1),('ordinary',day.y<1),('late',day.day>=179)]:
            g=day[(day.seed==s)&m]
            rows.append(dict(seed=s,segment=seg,n_days=len(g),bias_baseline=float((g.baseline-g.y).mean()),bias_arithmetic=float((g.candidate-g.y).mean()),bias_geometric=float((g.geometric-g.y).mean()),mean_inverse_change=float((g.candidate-g.geometric).mean())))
    result=dict(status='PASS',scope='Same trained log forest/actual baseline/mix/clip geometric-vs-arithmetic mechanism and loss-path diagnostic, not a new submitted candidate',
                rows=len(o),clip_counts=clip_counts,cells=cells,day_bias=rows,
                favourable_main_direction=sum(c['direction_derivative_at_zero']<0 for c in cells if c['validator'] in ['DIAG10','A','B']),main_direction_denominator=9,
                direction_definition='Stored final arithmetic candidate minus baseline; no oracle weight selected',
                raw_inverse_uplift_quantiles=np.quantile(o.new_et_raw-o.old_log_raw,[0,.5,.9,.99,1]).tolist(),
                source_sha256=S.sha(H/'run.py'),result_sha256=S.sha(H/'result_v1.json'))
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
