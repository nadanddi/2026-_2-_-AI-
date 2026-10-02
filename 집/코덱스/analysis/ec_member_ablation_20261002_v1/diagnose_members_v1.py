"""완료 OOF의 멤버별/제거안별 오차분해. 사후 설명용, 선택/튜닝에 쓰지 않음."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys,csv,json,math,importlib.util
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_member_ablation_20261002_v1'
PHASE=ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1'
ARMS=['drop_et','drop_lgb','drop_mlp','drop_pfn']
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load_json(path):return json.loads(path.read_text(encoding='utf-8'))
def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert (HERE/'completion.json').exists(),'Original experiment must complete first'
    assert load_json(HERE/'verification.json')['status']=='PASS','Independent arithmetic must pass first'
    spec=importlib.util.spec_from_file_location('ablation_independent_transform_readonly',HERE/'verify_ablation_v1.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    locks={(r['farm'],int(r['day'])) for r in load_json(Path(env.CODEX)/'ec_final_lock/locked_days.json')['selected']}
    truth={}
    for r in read(Path(env.DATA)/'train_y.csv'):
        f,d,_=r['row_id'].split('_')
        if f not in ['F13','F47'] or (f,int(d)) in locks:continue
        truth[r['row_id']]=float(r['sub_ec'])
    buckets=defaultdict(lambda:{'n':0,'day_occurrences':0,'sse':[],'level_sse':[],'error_sum':[],
                                'outside_train_range':0,'fold_rmse':[]})
    def add(name,seed,fold,model,mode,ids,y,p,lo,hi):
        farms=np.array([rid[:3] for rid in ids]);days=np.array([int(rid.split('_')[1]) for rid in ids])
        masks={'all':np.ones(len(ids),dtype=bool),'F13':farms=='F13','F47':farms=='F47','early':days<179,'late':days>=179}
        p=np.asarray(p,float);y=np.asarray(y,float)
        for segment,mask in masks.items():
            if not mask.any():continue
            e=p[mask]-y[mask];selected=np.flatnonzero(mask)
            groups=defaultdict(list)
            for j,pos in enumerate(selected):groups[(farms[pos],int(days[pos]))].append(float(e[j]))
            assert all(len(v)==24 for v in groups.values()),'Segments must contain complete daily trajectories'
            level=math.fsum(len(v)*(math.fsum(v)/len(v))**2 for v in groups.values())
            sse=math.fsum(float(x)**2 for x in e)
            key=(name,seed,model,mode,segment)
            b=buckets[key];b['n']+=len(e);b['day_occurrences']+=len(groups)
            b['sse'].append(sse);b['level_sse'].append(level);b['error_sum'].append(math.fsum(e.tolist()))
            b['outside_train_range']+=int(((p[mask]<lo)|(p[mask]>hi)).sum())
            b['fold_rmse'].append(math.sqrt(sse/len(e)))
    for a in load_json(HERE/'reconstruction_audit.json'):
        name=a['validator'];fold=int(a['validation_fold']);seed=int(a['seed']);lo=a['train_label_min'];hi=a['train_label_max']
        with np.load(OUT/f'{name}_{fold}_seed{seed}_members.npz') as z:
            ids=z['row_id'].tolist();members={k:z[k].copy() for k in ['et','lgb','mlp']}
        assert all(rid in truth for rid in ids)
        with np.load(PHASE/f'{name}_{fold}.npz') as z:
            assert ids==z['row_id'].tolist()
            r=z[f'r3_{seed}'];v=z[f'v2_{seed}'];finished=(v-.8*r)/.2
        assert a['cache_pfn_safe'] and a['clip_r3_count']==a['clip_v2_count']==0
        members['pfn']=np.array(verifier.transform(ids,finished,inverse=True))
        target=np.array([truth[rid] for rid in ids])
        for model,p in members.items():
            add(name,seed,fold,model,'raw_no_clip',ids,target,p,lo,hi)
            q=verifier.transform(ids,p)
            add(name,seed,fold,model,'causal_shrink_no_clip',ids,target,q,lo,hi)
    groups=defaultdict(list)
    for r in read(OUT/'oof_predictions.csv'):
        assert abs(float(r['sub_ec'])-truth[r['row_id']])<1e-12
        groups[(r['validator'],int(r['validation_fold']),int(r['seed']))].append(r)
    for (name,fold,seed),rows in groups.items():
        a=next(a for a in load_json(HERE/'reconstruction_audit.json') if (a['validator'],int(a['validation_fold']),int(a['seed']))==(name,fold,seed))
        ids=[r['row_id'] for r in rows];target=[truth[rid] for rid in ids]
        for arm in ['v2']+ARMS:
            add(name,seed,fold,arm,'final_shrink_clip',ids,target,[float(r[arm]) for r in rows],a['train_label_min'],a['train_label_max'])
    output=[]
    for (name,seed,model,mode,segment),b in sorted(buckets.items()):
        sse=math.fsum(b['sse']);level=math.fsum(b['level_sse']);shape=sse-level
        assert shape>-1e-10 and b['n']==b['day_occurrences']*24
        output.append({'validator':name,'seed':seed,'model':model,'processing':mode,'segment':segment,
                       'n_rows':b['n'],'day_occurrences':b['day_occurrences'],'rmse':math.sqrt(sse/b['n']),
                       'level_rmse':math.sqrt(level/b['n']),'shape_rmse':math.sqrt(max(0.,shape)/b['n']),
                       'level_error_share':level/sse,'mean_error':math.fsum(b['error_sum'])/b['n'],
                       'outside_train_range':b['outside_train_range'],'fold_rmse_mean':float(np.mean(b['fold_rmse'])),
                       'fold_rmse_std':float(np.std(b['fold_rmse'],ddof=1)) if len(b['fold_rmse'])>1 else ''})
    save_csv(HERE/'member_diagnostics.csv',[r for r in output if r['model'] in ['et','lgb','mlp','pfn']])
    save_csv(HERE/'ablation_error_diagnostics.csv',[r for r in output if r['model'] in ['v2']+ARMS])
    summary={'status':'COMPLETE','diagnostic_only':True,'used_for_weight_selection':False,
             'individual_member_clip_applied':False,'level_plus_shape_sse_identity_passed':True,
             'member_diagnostic_rows':sum(r['model'] in ['et','lgb','mlp','pfn'] for r in output),
             'ablation_diagnostic_rows':sum(r['model'] in ['v2']+ARMS for r in output),
             'final_lock_scored':False,'new_model_fitted':False,
             'important_limit':'A/B days may recur across folds; n and daily counts are OOF occurrence denominators.'}
    (HERE/'diagnostics_completion.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
