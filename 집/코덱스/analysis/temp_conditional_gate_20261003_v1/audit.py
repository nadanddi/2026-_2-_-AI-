from pathlib import Path
import json,csv,math,collections
from temp_conditional_gate_model import m,GATE_COLS,np
H=Path(__file__).resolve().parent;O=H.parents[1]/'local/temp_conditional_gate_20261003_v1'
result=json.loads((O/'result.json').read_text(encoding='utf-8'));z=dict(np.load(Path(m.env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True));zi={str(q):i for i,q in enumerate(z['row_id'])}
with (Path(m.env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:labels={x['row_id']:float(x['sub_temp']) for x in csv.DictReader(f)}
tx,_,sx=m.TM.masked_loader();features=m.build_features(tx,sx).set_index('row_id');audits=[];weight_rows=[]
def key(q):return q[:3],int(q[4:7])
for name,nfold in [('DIAG10',10),('EXT10',1),('EXT12',1)]:
    pfs=[np.load(Path(m.env.LOCAL)/f'web_tabpfn_v2_temp_{name}.npy').mean(0),np.load(Path(m.env.LOCAL)/f'web_tabpfn_v6_temp_{name}.npy')[:8].mean(0)]
    for k in range(nfold):
        saved=dict(np.load(O/f'{name}_{k}.npz',allow_pickle=True));stats=json.loads((O/f'{name}_{k}_training.json').read_text(encoding='utf-8'));valid={key(q) for q in saved['row_id']};ix=[zi[q] for q in saved['row_id']]
        for seed in [726,727]:
            ids=saved[f'cal_row_id_{seed}'];train={key(q) for q in ids};assert not train&valid and all(f!=g or abs(d-e)>1 for f,d in train for g,e in valid);st=stats[str(seed)]['G1'];seen=set()
            for inn in st['inner_split_audit']:
                t={tuple(q) for q in inn['training_days']};v={tuple(q) for q in inn['query_days']};forbidden={(f,d+j) for f,d in v for j in [-1,0,1]};assert t==train-forbidden and v<=train and not seen&v;seen|=v
            assert seen==train
            y=saved[f'cal_y_{seed}'];w=saved[f'cal_w_{seed}'];b=saved[f'cal_BASE_{seed}'];c=saved[f'cal_CODEX_{seed}'];x=saved[f'cal_design_{seed}'];target=saved[f'cal_target_{seed}'];assert all(labels[q]==float(v) for q,v in zip(ids,y))
            coef=np.linalg.solve(x.T@(w[:,None]*x)+100*np.eye(x.shape[1]),x.T@(w*target));coefdiff=float(np.max(np.abs(coef-np.array(st['gate_coef']))));assert coefdiff<1e-10
            ft=features.loc[ids];fv=features.loc[saved['row_id']];gt=np.where(np.isnan(ft.in_temp),1.,np.clip((ft.in_temp.to_numpy()-8)/2,0,1));gv=np.where(np.isnan(fv.in_temp),1.,np.clip((fv.in_temp.to_numpy()-8)/2,0,1));beta=(.4+.1*gt)/(1-.3*gt);assert np.max(np.abs(target-(y-beta*b-(1-beta)*c)))<1e-10
            # Recreate preprocessing independently from the raw causal feature columns.
            arr=ft[GATE_COLS].to_numpy(float);av=fv[GATE_COLS].to_numpy(float);med=np.nanmedian(arr,axis=0);med=np.nan_to_num(med);arr=np.where(np.isnan(arr),med,arr);av=np.where(np.isnan(av),med,av);mu=arr.mean(0);sd=arr.std(0);sd[sd==0]=1;zt=np.column_stack([np.ones(len(arr)),(arr-mu)/sd]);zv=np.column_stack([np.ones(len(av)),(av-mu)/sd]);dx=float(np.max(np.abs(x-zt*(gt*(b-c))[:,None])));assert dx<1e-9
            dt=gt*np.clip(zt@coef,-.15,.15);dv=gv*np.clip(zv@coef,-.15,.15);assert np.max(np.abs(dt-saved[f'cal_delta_{seed}']))<1e-10 and np.max(np.abs(dv-saved[f'gate_delta_{seed}']))<1e-10
            bs={726:7,727:101}[seed];ob=z[f'{name}__MASK__{bs}'][ix];oc=z[f'{name}__CODEX__{seed}'][ix];assert np.max(np.abs(ob-saved[f'outer_BASE_{seed}']))==0 and np.max(np.abs(oc-saved[f'outer_CODEX_{seed}']))==0
            wb=.4+.1*gv+(1-.3*gv)*dv;wc=.6-.4*gv-(1-.3*gv)*dv;wp=.3*gv;assert np.min(wb)>=0 and np.min(wc)>=0 and np.max(np.abs(wb+wc+wp-1))<1e-12
            for tag,p in [('BASE',wb),('CODEX',wc),('PFN',wp)]:assert np.max(np.abs(p-saved[f'w_{tag}_{seed}']))<1e-10
            before=(.4+.1*gv)*ob+(.6-.4*gv)*oc;after=wb*ob+wc*oc;assert np.max(np.abs(before[gv==0]-after[gv==0]),initial=0)<1e-12
            for pf in pfs:
                direct=after+wp*pf[ix];compatible=(.4+.1*gv)*ob+(.6-.4*gv)*saved[f'G1_{seed}']+wp*pf[ix];assert np.max(np.abs(direct-compatible))<1e-12
            audits.append(dict(validator=name,fold=k,seed=seed,coef_maxdiff=coefdiff,design_maxdiff=dx,inner_split='PASS',cold_invariance='PASS',sum_to_one='PASS'))
            if name=='DIAG10':
                for q,bb,cc,pp in zip(saved['row_id'],wb,wc,wp):weight_rows.append(dict(seed=seed,row_id=q,farm=q[:3],hour=int(q[8:10]),in_temp=float(features.loc[q,'in_temp']),w_BASE=float(bb),w_CODEX=float(cc),w_PFN=float(pp)))
segments=[]
for seed in [726,727]:
    for segment,predicate in [('all',lambda x:True),('cold',lambda x:x['in_temp']<=8),('warm',lambda x:x['in_temp']>=10),('night',lambda x:x['hour']<6),('day',lambda x:6<=x['hour']<18),('evening',lambda x:x['hour']>=18),('F13',lambda x:x['farm']=='F13'),('F47',lambda x:x['farm']=='F47')]:
        rr=[q for q in weight_rows if q['seed']==seed and predicate(q)];n=len(rr);assert n
        segments.append(dict(seed=seed,segment=segment,n=n,**{f'{name}_{stat}':fn([q[name] for q in rr]) for name in ['w_BASE','w_CODEX','w_PFN'] for stat,fn in [('min',min),('max',max),('mean',lambda a:math.fsum(a)/len(a))]}))
out=dict(status='PASS',audits=audits,weight_segments=segments)
(H/'gate_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({'status':'PASS','max_coef_diff':max(q['coef_maxdiff'] for q in audits),'weights':segments[:3]}))
