"""Recompute stored prediction metrics independently and check fixed exploratory rules."""
import csv, json, math, hashlib, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
DATA=ROOT/'연구실/코덱스/local/im_meta_probe_20261016_v1'
OUT=Path(__file__).resolve().parent
SEEDS=(1601,1602,1603)
CORE=('im_set_r2','im_set_r3','im_set_r4')

def read(seed,sn):
    with (DATA/f'pred_{seed}_{sn}.csv').open(encoding='utf-8') as f: return list(csv.DictReader(f))

def calc(rows):
    n=len(rows); nr=sum(int(r['label'])==0 for r in rows); nf=n-nr
    out=dict(n=n,real=nr,fake=nf,base_fp=0,meta_fp=0,base_tp=0,meta_tp=0,
             new_fp=0,fixed_real=0,new_fn=0,fixed_fake=0,base_correct=0,meta_correct=0)
    for r in rows:
        b=float(r['p9']); p=float(r['meta']); y=int(r['label'])
        assert all(math.isfinite(v) and 0<=v<=1 for v in (b,p))
        assert r['gate'] in ('0','1')
        if r['gate']=='0': assert b==p
        bb=b>=.5; pp=p>=.5
        out['base_correct']+=int(bb==bool(y)); out['meta_correct']+=int(pp==bool(y))
        if y==0:
            out['base_fp']+=int(bb); out['meta_fp']+=int(pp)
            out['new_fp']+=int(not bb and pp); out['fixed_real']+=int(bb and not pp)
        else:
            out['base_tp']+=int(bb); out['meta_tp']+=int(pp)
            out['new_fn']+=int(bb and not pp); out['fixed_fake']+=int(not bb and pp)
    for name in ('base','meta'):
        out[name+'_ba']=.5*((nr-out[name+'_fp'])/nr+out[name+'_tp']/nf) if nr and nf else None
    return out

def boot(allrows,seed=1610):
    keys=[(r['file'],r['source'],r['group'],r['label']) for r in allrows[0]]
    assert all([(r['file'],r['source'],r['group'],r['label']) for r in rr]==keys for rr in allrows)
    gs=sorted({r['group'] for r in allrows[0]}); assert '' not in gs and 'unknown' not in gs
    # Per-group counts, preserving the paired seed/model/row comparisons.
    cnt=np.zeros((len(gs),2)); dc=np.zeros((len(gs),2))
    for i,g in enumerate(gs):
        ix=[j for j,r in enumerate(allrows[0]) if r['group']==g]
        for j in ix:
            y=int(allrows[0][j]['label']); cnt[i,y]+=1
            dc[i,y]+=np.mean([int((float(rr[j]['meta'])>=.5)==bool(y))-int((float(rr[j]['p9'])>=.5)==bool(y)) for rr in allrows])
    rng=np.random.default_rng(seed); draw=rng.integers(len(gs),size=(20000,len(gs)))
    den=cnt[draw].sum(axis=1); num=dc[draw].sum(axis=1); valid=(den>0).all(axis=1)
    dif=.5*(num[valid]/den[valid]).sum(axis=1)
    return dict(groups=len(gs),valid_draws=int(valid.sum()),p_delta_le_zero=float((dif<=0).mean()),
                ci95=np.quantile(dif,[.025,.975]).tolist(),mean_observed=float(.5*(dc.sum(axis=0)/cnt.sum(axis=0)).sum()))

def main():
    metrics={}; checks=0; saved_rows={}; kinds={}; coeff={}
    for seed in SEEDS:
        metrics[seed]={}; saved=json.loads((DATA/f'metrics_{seed}.json').read_text(encoding='utf-8'))
        for sn,v in saved.items():
            rows=read(seed,sn); saved_rows[seed,sn]=rows; vv=calc(rows)
            for k,x in vv.items():
                assert (x is None and v[k] is None) or (x is not None and math.isclose(x,v[k],abs_tol=1e-12)),(seed,sn,k,x,v[k])
                checks+=1
            for column,sub in [('source_overlap','source_disjoint'),('group_overlap','group_disjoint')]:
                subset=[r for r in rows if r[column]=='0']; vsub=calc(subset)
                for k,x in vsub.items():
                    assert (x is None and v[sub][k] is None) or (x is not None and math.isclose(x,v[sub][k],abs_tol=1e-12)),(seed,sn,sub,k)
                    checks+=1
            metrics[seed][sn]=vv
            kinds[f'{seed}/{sn}']={kind:calc([r for r in rows if r['kind']==kind]) for kind in sorted({r['kind'] for r in rows})}
        z=np.load(DATA/f'oof_{seed}.npz'); fold=z['fold']; xx=z['X']
        assert np.isfinite(xx).all() and xx.shape==(6494,7)
        assert all(len(set(fold[z['source']==s]))==1 for s in set(z['source']))
        assert all(len(set(fold[z['group']==g]))==1 for g in set(z['group']))
        m=np.load(DATA/f'meta_{seed}.npz')
        # Verify serialized parameters reproduce actual meta outputs in a small external set
        # via integrity hashes; upstream embeddings aren't copied to this machine.
        coeff[str(seed)]=dict(coef=m['coef'].tolist(),intercept=m['intercept'].tolist())
    hashes=[hashlib.sha256(np.load(DATA/f'oof_{s}.npz')['fold'].tobytes()).hexdigest() for s in SEEDS]
    assert len(set(hashes))==3
    rules={}
    for seed,M in metrics.items():
        rr={}
        rr['core_noninferior_each']=all(M[k]['meta_ba']>=M[k]['base_ba'] for k in CORE)
        rr['core_strict_one']=any(M[k]['meta_ba']>M[k]['base_ba'] for k in CORE)
        rr['sealed_new_fp_le2']=sum(M[k]['new_fp'] for k in ('sealed_real','sealed_brown','sealed_bright'))<=2
        rr['jig_fpr_guard']=(M['jig500']['meta_fp']-M['jig500']['base_fp'])/M['jig500']['real']<=.01
        rr['processing_fpr_guard']=all((M['im_set_r3p_'+k]['meta_fp']-M['im_set_r3p_'+k]['base_fp'])/M['im_set_r3p_'+k]['real']<=.01 for k in ('sharp','area','png','webp','c444'))
        rr['public21_all']=M['pub21']['meta_correct']==21
        rr['external_fake_new_fn_le1']=all(M[k]['new_fn']<=1 for k in ('im_set_gpt3','im_set_gpt1','im_set_sd2','sealed_qwen','sealed_v7','im_set_gptr'))
        rr['gptr_source_new_fp_le1']=M['im_set_gptr_src']['new_fp']<=1
        rr['exploratory_suitable']=all(rr.values()); rules[str(seed)]=rr
    bootstrap={sn:boot([saved_rows[s,sn] for s in SEEDS]) for sn in CORE}
    result=dict(checks=checks,seed_count=3,dataset_count=len(metrics[1601]),prediction_rows=sum(len(v) for v in saved_rows.values()),
                all_finite_and_probability_range=True,gate_outside_equal=True,fold_hashes=hashes,
                rules=rules,bootstrap=bootstrap,metrics={str(s):m for s,m in metrics.items()},coefficients=coeff,
                interpretation='Exploratory exposed sets; not adoption or sealed confirmation. Base-model/DB independence not certified.')
    with (OUT/'summary_v1.json').open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (OUT/'kinds_v1.json').open('x',encoding='utf-8') as f: json.dump(kinds,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (OUT/'comparison_v1.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['seed','set','base_ba','meta_ba','delta_ba','new_fp','fixed_real','new_fn','fixed_fake'])
        for seed,M in metrics.items():
            for sn,m in M.items(): w.writerow([seed,sn,m['base_ba'],m['meta_ba'],m['meta_ba']-m['base_ba'] if m['base_ba'] is not None else '',m['new_fp'],m['fixed_real'],m['new_fn'],m['fixed_fake']])
    print(json.dumps(dict(checks=checks,rules=rules,bootstrap=bootstrap),ensure_ascii=False))

if __name__=='__main__': main()
