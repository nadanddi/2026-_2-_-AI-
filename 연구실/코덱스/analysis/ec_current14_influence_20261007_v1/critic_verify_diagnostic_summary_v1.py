"""Independent diagnostic arithmetic and parent support-audit receipt review."""
from pathlib import Path
import json,csv,math,hashlib,collections
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'연구실/코덱스/local'/H.name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=2e-10),(a,b)
d=read(H/'ablation_score_v1/days.csv');lookup={(r['arm'],r['seed'],r['farm'],int(r['day'])):r for r in d}
summary=json.loads((H/'diagnostic_summary_v1.json').read_text(encoding='utf-8'));assert summary['source_sha']==sha(H/'diagnostic_summary_v1.py') and summary['source_scores_sha']==sha(H/'ablation_score_v1/completion.json')
for name,s in summary['files'].items():assert sha(H/name)==s
expected={}
for r in d:
    if r['arm'] not in ['D1','D2']:continue
    b=lookup[('BASE',r['seed'],r['farm'],int(r['day']))];key=(r['arm'],r['seed'],r['farm'],int(r['day']))
    expected[key]=dict(delta_sse=float(r['sse'])-float(b['sse']),prediction_change=float(r['prediction'])-float(b['prediction']),raw_et_change=float(r['raw_et'])-float(b['raw_et']))
changes=read(H/'day_changes_v1.csv');assert len(changes)==2880
for r in changes:
    key=(r['arm'],r['seed'],r['farm'],int(r['day']));source=lookup[key]
    for n,v in expected[key].items():near(v,r[n])
    for n in ['truth','prediction','sse','raw_et','rmse']:near(source[n],r[n])
for s in summary['summaries']:
    u=[(k,v) for k,v in expected.items() if k[0]==s['arm'] and k[1]==s['seed']];ordinary=[v['delta_sse'] for k,v in u if float(lookup[k]['truth'])<1];gain=-math.fsum(ordinary);selected=-expected[(s['arm'],s['seed'],'F47',161)]['delta_sse'];remaining=math.fsum(v['delta_sse'] for k,v in u if float(lookup[k]['truth'])<1 and (k[2],k[3])!=('F47',161))
    near(gain,s['ordinary_net_sse_gain']);near(selected,s['F47_161_sse_gain']);near(remaining,s['ordinary_remaining_delta_sse']);near(selected/gain,s['selected_fraction_of_ordinary_net_gain'])
    for n,val in [('improved_days',sum(v['delta_sse'] < -1e-12 for _,v in u)),('worse_days',sum(v['delta_sse']>1e-12 for _,v in u)),('unchanged_days',sum(abs(v['delta_sse'])<=1e-12 for _,v in u))]:assert s[n]==val
    near(math.fsum(v['delta_sse'] for _,v in u if v['delta_sse']>0),s['positive_loss_sum']);near(math.fsum(v['delta_sse'] for _,v in u if v['delta_sse']<0),s['negative_loss_sum'])
for name,reverse in [('largest_losses_v1.csv',True),('largest_gains_v1.csv',False)]:
    r=read(H/name);assert len(r)==24
    for arm in ['D1','D2']:
        ordered=sorted([(k,v) for k,v in expected.items() if k[0]==arm and k[1]=='ensemble'],key=lambda kv:kv[1]['delta_sse'],reverse=reverse)[:12]
        assert {(x['farm'],int(x['day'])) for x in r if x['arm']==arm}=={(k[2],k[3]) for k,v in ordered}
support=read(H/'influence_support_days_v1.csv');index={(r['arm'],r['level'],r['farm'],int(r['day'])):r for r in support}
labels={}
for r in read(ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv'):
    if r['validator']=='DIAG10':labels[r['row_id']]=float(r['y'])
yl=collections.defaultdict(list)
for rid,y in labels.items():yl[(rid[:3],int(rid[4:7]))].append(y)
ym={k:math.fsum(v)/len(v) for k,v in yl.items()}
profiles=read(H/'influence_support_profiles_v1.csv');assert len(profiles)==9
for p in profiles:
    g=[r for r in support if r['arm']==p['arm'] and r['level']==p['level']]
    near(math.fsum(float(r['weight']) for r in g),1);near(math.fsum(float(r['contribution']) for r in g),p['prediction']);near(math.fsum(float(r['weight']) for r in g if ym[(r['farm'],int(r['day']))]>=1),p['high_day_weight'])
    for r in g:near(float(r['contribution'])/float(r['weight']),r['weighted_ec'])
shifts=read(H/'support_shifts_v1.csv')
for r in shifts:
    farm,day=r['farm'],int(r['day']);b=index.get(('BASE','smooth_day',farm,day));a=index.get((r['arm'],'smooth_day',farm,day))
    for name,col in [('weight','weight'),('contribution','contribution')]:
        bv=float(b[col]) if b else 0;av=float(a[col]) if a else 0;near(bv,r['base_'+name]);near(av,r['arm_'+name]);near(av-bv,r['delta_'+name])
parent=json.loads((H/'support_audit_v1.json').read_text(encoding='utf-8'));assert parent['status']=='PASS_ALL_SAVED_TREE_PATHS_AND_SUPPORT' and parent['new_fit']==0 and parent['source_sha']==sha(H/'audit_support_v1.py')
assert parent['total_nodes']==sum(m['nodes'] for m in parent['models'])==20321234 and sum(m['trees'] for m in parent['models'])==1800
for m in parent['models']:
    meta=json.loads((L/'trace'/f"{m['arm']}.json").read_text(encoding='utf-8'));assert m['sha']==meta['sha']==sha(L/'trace'/f"{m['arm']}.npz") and m['nodes']==meta['nodes'] and m['query_rows']==24 and m['trees']==600 and m['node_mean_maxdiff']<1e-10 and m['replay_maxdiff']<1e-10
important={}
for arm in ['BASE','D1','D2']:
    p=next(r for r in profiles if r['arm']==arm and r['level']=='smooth_day');important[arm]=dict(smooth_high_day_weight=float(p['high_day_weight']),smooth_ET=float(p['prediction']))
    g=sorted([r for r in support if r['arm']==arm and r['level']=='smooth_day'],key=lambda r:float(r['weight']),reverse=True);important[arm]['top_support']=[dict(farm=r['farm'],day=int(r['day']),weight=float(r['weight']),weighted_ec=float(r['weighted_ec'])) for r in g[:5]]
out=dict(status='PASS_DIAGNOSTIC_ARITHMETIC_AND_PARENT_SUPPORT_RECEIPT_REVIEW',fit=0,day_changes=2880,summary_cells=8,support_profiles=9,largest_changes=48,important_support=important,parent_audit_trees=1800,parent_audit_nodes=20321234,parent_audit_ownership='Parent wrote/executed routing/count/mean/weights audit; critic code+receipt/hash review only, no duplicate tree routing.',ratio_warning='selected_fraction with a negative net gain is a signed arithmetic ratio, not a share of improvement.')
with (H/'critic_verify_diagnostic_summary_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
