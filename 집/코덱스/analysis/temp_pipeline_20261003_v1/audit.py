from pathlib import Path
import sys,json,csv,math,collections
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
O=ROOT/'집/코덱스/local/temp_pipeline_20261003_v1'
r=json.loads((O/'result.json').read_text(encoding='utf-8'));z=dict(np.load(Path(env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True));zi={str(q):i for i,q in enumerate(z['row_id'])}
with (Path(env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:labels={x['row_id']:float(x['sub_temp']) for x in csv.DictReader(f)}
records=[];baseline_diff=0.;fold_stats=[]
def daykey(q):return q[:3],int(q[4:7])
for name,nfold in [('DIAG10',10),('EXT10',1),('EXT12',1)]:
    for k in range(nfold):
        c=dict(np.load(O/f'{name}_{k}.npz',allow_pickle=True));info=json.loads((O/f'{name}_{k}_training.json').read_text(encoding='utf-8'));valid={daykey(q) for q in c['row_id']}
        for seed in [726,727]:
            original=z[f'{name}__CODEX__{seed}'][[zi[str(q)] for q in c['row_id']]];bd=float(np.max(np.abs(original-c[f'original_CODEX_{seed}'])));baseline_diff=max(baseline_diff,bd);assert bd<1e-10
            ids=c[f'cal_row_id_{seed}'];y=c[f'cal_y_{seed}'];w=c[f'cal_w_{seed}'];p=c[f'cal_physics_{seed}'];a=c[f'cal_residual_{seed}'];assert all(float(v)==labels[q] for q,v in zip(ids,y));train={daykey(q) for q in ids};assert not train&valid;assert all(f!=g or abs(d-e)>1 for f,d in train for g,e in valid)
            numerator=math.fsum(float(ww)*float(aa)*(float(yy)-float(pp)) for ww,aa,yy,pp in zip(w,a,y,p))+100.;denominator=math.fsum(float(ww)*float(aa)**2 for ww,aa in zip(w,a))+100.;gamma=max(0.,min(1.25,numerator/denominator));st=info[str(seed)]['I1'];assert abs(gamma-st['gamma'])<1e-12 and abs(numerator-st['numerator'])<1e-9 and abs(denominator-st['denominator'])<1e-9
            seen=set()
            for inner in info[str(seed)]['A1']['inner_split_audit']:
                t={tuple(q) for q in inner['training_days']};v={tuple(q) for q in inner['query_days']};assert t<=train and v<=train and not t&v and not v&seen;assert all(f!=g or abs(d-e)>1 for f,d in t for g,e in v);seen|=v
                forbidden={(f,d+j) for f,d in v for j in [-1,0,1]};assert t==train-forbidden
            assert seen==train
            reconstructed=c[f'physics_{seed}']+gamma*(c[f'original_CODEX_{seed}']-c[f'physics_{seed}']);delta=float(np.max(np.abs(reconstructed-c[f'I1_{seed}'])));assert delta<1e-12
            records.append(dict(validator=name,fold=k,seed=seed,gamma=gamma,train_days=len(train),valid_days=len(valid),original_CODEX_maxdiff=bd,inference_formula_maxdiff=delta))
with (O/'oof.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
for tag in ['A1','B1','C1','I1']:
    for seed in ['726','727']:
        for ctx in ['1-8','17-24']:
            scores=[];deltas=[]
            for k in range(10):
                rr=[x for x in rows if x['member']==tag and x['seed']==seed and x['context']==ctx and x['validator']=='DIAG10' and int(x['fold'])==k];n=len(rr);b=math.sqrt(math.fsum((float(x['base'])-float(x['sub_temp']))**2 for x in rr)/n);a=math.sqrt(math.fsum((float(x['candidate'])-float(x['sub_temp']))**2 for x in rr)/n);scores.append(a);deltas.append(a-b)
            mu=math.fsum(scores)/10;md=math.fsum(deltas)/10;fold_stats.append(dict(member=tag,seed=seed,context=ctx,fold_mean=mu,fold_sd=math.sqrt(math.fsum((v-mu)**2 for v in scores)/9),paired_delta_sd=math.sqrt(math.fsum((v-md)**2 for v in deltas)/9),improved_folds=sum(v<0 for v in deltas)))
answer=dict(status='PASS',original_CODEX_maxdiff=baseline_diff,inner_and_outer_split_and_gamma=records,fold_statistics=fold_stats)
(H/'pipeline_audit.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':'PASS','original_CODEX_maxdiff':baseline_diff,'gamma_range':[min(s['gamma'] for s in records),max(s['gamma'] for s in records)]}))
