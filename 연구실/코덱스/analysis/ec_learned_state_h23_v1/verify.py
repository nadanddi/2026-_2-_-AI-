"""Independent stdlib arithmetic/provenance audit of a completed H23 run."""
import csv,hashlib,json,math,sys
from pathlib import Path
from collections import defaultdict
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
out=Path(sys.argv[1]);result=json.loads((out/'result.json').read_text(encoding='utf-8'))
def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(a,b)
def rmse(rows,name):return math.sqrt(sum((float(r['sub_ec'])-float(r[name]))**2 for r in rows)/len(rows))
def near(days):return {(str(f),int(d)+k) for f,d in days for k in (-1,0,1)}
hash_count=0
for p,want in result['sha256'].items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==want,p
    hash_count+=1
allrows=defaultdict(list)
for score in result['scores']:
    f,s=score['fold'],score['seed']
    with (out/f'fold{f}_seed{s}.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    assert len({r['row_id'] for r in rows})==len(rows)
    assert len(rows)%24==0
    for c in ('v2','direct','candidate','base','direct_et','state_et'):close(rmse(rows,c),score[c])
    # Independently reconstruct causal .5 current + .5 prefix mean corrections.
    groups=defaultdict(list)
    for r in rows:groups[(r['farm'],int(r['day']))].append(r)
    for g in groups.values():
        g.sort(key=lambda r:int(r['hour']));assert [int(r['hour']) for r in g]==list(range(24))
        for name,et in [('direct','direct_et'),('candidate','state_et')]:
            running=0
            for k,r in enumerate(g,1):
                delta=float(r[et])-float(r['base']);running+=delta
                unclip=float(r['v2'])+.24*(.5*delta+.5*running/k)
                # Bounds are supplied by training labels below, not inferred from prediction extrema.
                r['_unclip_'+name]=unclip
    allrows[s].extend(rows)
audits=json.loads((out/'training_audit.json').read_text(encoding='utf-8'))
lock=json.loads((ROOT/'집/코덱스/analysis/codex_independent/ec_final_lock/locked_days.json').read_text(encoding='utf-8'))
locked={(v['farm'],int(v['day'])) for v in lock['selected']}
hold=set()
for a in audits:
    if a['fold'] in (8,9):hold.update(map(tuple,a['held_days']))
with (ROOT/'공용/대회자료/정형데이터/참가자_배포/train_y.csv').open(encoding='utf-8',newline='') as handle:
    target={r['row_id']:float(r['sub_ec']) for r in csv.DictReader(handle) if r['row_id'].startswith(('F13_','F47_')) and r['sub_ec']}
for a in audits:
    train=set(map(tuple,a['train_days']));held=set(map(tuple,a['held_days']))
    assert not train&near(held|locked|hold)
    values=[v for rid,v in target.items() if (rid[:3],int(rid[4:7])) in train]
    lo,hi=min(values),max(values)
    for r in allrows[a['seed']]:
        if int(r['fold'])!=a['fold']:continue
        close(float(r['sub_ec']),target[r['row_id']])
        for name in ('candidate','direct'):close(float(r[name]),min(hi,max(lo,r['_unclip_'+name])))
    for inner in a['inner']:
        it=set(map(tuple,inner['train_days']));ih=set(map(tuple,inner['held_days']))
        assert it<=train and ih<=train and not it&near(ih)
    assert set().union(*(set(map(tuple(i['held_days'])) for i in a['inner'])) )==train
for seed,rows in allrows.items():
    assert len(rows)==5616 and len({r['row_id'] for r in rows})==5616
    s=result['per_seed'][str(seed)]
    for c in ('v2','direct','candidate','base','direct_et','state_et'):close(rmse(rows,c),s[c])
    for c in ('v2','direct'):
        delta=sum((float(r['sub_ec'])-float(r['candidate']))**2-(float(r['sub_ec'])-float(r[c]))**2 for r in rows)/len(rows)
        close(delta,s['comparisons'][c]['delta_mse'])
    aux=[]
    for fold in (0,2,4,6,8,9):
        with (out/f'aux_fold{fold}_seed{seed}.csv').open(encoding='utf-8',newline='') as handle:aux.extend(csv.DictReader(handle))
    for n in ('dah','dco2'):
        a=[r for r in aux if r[n] and int(r['hour'])>0]
        e=sum((float(r[n])-float(r['pred_'+n]))**2 for r in a)/len(a)
        zero=sum(float(r[n])**2 for r in a)/len(a)
        close(e**.5,s['aux'][n]['rmse']);close(zero**.5,s['aux'][n]['persistence_rmse'])
        close(1-e/zero,s['aux'][n]['skill_vs_zero'])
assert result['final_lock_scored'] is False and result['test_X_read'] is False and result['submission_created'] is False
verified={'status':'PASS','hash_count':hash_count,'outer_seed_audits':len(audits),'rows_per_seed':5616,'checks':['raw labels','fold and pooled RMSE','causal candidate arithmetic and clipping','auxiliary skill','outer/inner purge and coverage','hashes'],'bootstrap_recalculated':False}
(out/'verification.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
print(json.dumps(verified))
