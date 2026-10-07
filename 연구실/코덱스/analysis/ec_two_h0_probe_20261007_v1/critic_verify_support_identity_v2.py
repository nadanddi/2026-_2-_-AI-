"""Trace/parent receipt and support-input identity arithmetic, no tree rerouting."""
from pathlib import Path
import sys,json,csv,math,hashlib,collections
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as R
np=R.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-10),(a,b)
parent=json.loads((H/'tree_audit_v1.json').read_text(encoding='utf-8'));assert parent['status']=='PASS_FULL_600_TREE_PATHS_SUPPORT_AND_RETAINED_MEDIAN' and parent['trees']==600 and parent['new_fit']==0 and parent['source_sha']==sha(H/'audit_tree_v1.py') and parent['trace_sha']==sha(R.L/'trace.npz')
meta=json.loads((R.L/'trace.json').read_text(encoding='utf-8'));assert meta['sha']==parent['trace_sha'] and meta['source_sha']==sha(H/'run_v1.py') and meta['prep_sha']==sha(H/'preparation_v1.json') and (meta['arm'],meta['k'],meta['seed'])==(R.ARM,1,7) and meta['columns']==R.COLS and meta['nodes']==parent['nodes'] and parent['node_mean_maxdiff']<1e-10 and parent['ET_replay_maxdiff']<1e-10
labels={}
for r in read(R.ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv'):
    if r['validator']=='DIAG10':labels[r['row_id']]=float(r['y'])
ym=collections.defaultdict(list)
for rid,y in labels.items():ym[(rid[:3],int(rid[4:7]))].append(y)
ym={k:math.fsum(v)/len(v) for k,v in ym.items()}
with np.load(R.L/'trace.npz',allow_pickle=False) as z,np.load(R.B.L/'trace/BASE.npz',allow_pickle=False) as old,np.load(R.L/'1_7.npz',allow_pickle=False) as cache:
    assert z['columns'].tolist()==R.COLS and z['train_row_id'].tolist()==old['train_row_id'].tolist() and z['query_row_id'].tolist()==old['query_row_id'].tolist()
    assert all(labels[i]==y for i,y in zip(z['train_row_id'],z['train_y'])) and all(labels[i]==y for i,y in zip(z['query_row_id'],z['query_y']))
    keep=[R.B.M.FULL_R3.index(c) for c in R.COLS];assert np.array_equal(z['imputer_median'],old['imputer_median'][keep]) and np.array_equal(z['train_X'],old['train_X'][:,keep]) and np.array_equal(z['query_X'],old['query_X'][:,keep])
    index={i:j for j,i in enumerate(cache['row_id'])};assert np.max(abs(z['raw_et']-cache['et'][[index[i] for i in z['query_row_id']]]))<1e-10
    assert z['offsets'][-1]==parent['nodes'] and len(z['offsets'])==601
    w=z['weights'].copy();ty=z['train_y'].copy();ids=z['train_row_id'].copy();queryy=z['query_y'].copy()
profiles=read(H/'support_profiles_v1.csv');support=read(H/'support_days_v1.csv');assert len(profiles)==3 and {p['level'] for p in profiles}=={'hour0','raw_day','smooth_day'}
sw=np.array([.5*w[i]+.5*w[:i+1].mean(axis=0) for i in range(24)]);vectors={'hour0':w[0],'raw_day':w.mean(axis=0),'smooth_day':sw.mean(axis=0)}
profile_values={}
for p in profiles:
    assert p['arm']==R.ARM;v=vectors[p['level']];near(math.fsum(v),1);pred=math.fsum(float(a)*float(b) for a,b in zip(v,ty));hd=math.fsum(float(a) for a,i in zip(v,ids) if ym[(i[:3],int(i[4:7]))]>=1);hr=math.fsum(float(a) for a,y in zip(v,ty) if y>=1)
    near(pred,p['prediction']);near(hd,p['high_day_weight']);near(hr,p['high_row_weight']);near(queryy[0] if p['level']=='hour0' else math.fsum(queryy)/24,p['truth'])
    g=collections.defaultdict(list)
    for i,a,y in zip(ids,v,ty):g[(i[:3],int(i[4:7]))].append((float(a),float(a*y)))
    expected={k:(math.fsum(a for a,b in values),math.fsum(b for a,b in values)) for k,values in g.items() if math.fsum(a for a,b in values)>0};selected=[r for r in support if r['level']==p['level']]
    assert len(selected)==len(expected) and {(r['farm'],int(r['day'])) for r in selected}==set(expected)
    for r in selected:
        weight,contribution=expected[(r['farm'],int(r['day']))];near(weight,r['weight']);near(contribution,r['contribution']);near(contribution/weight,r['weighted_ec'])
    profile_values[p['level']]=dict(prediction=pred,high_day_weight=hd,high_row_weight=hr)
# Input-only independent raw csv confirmation. No labels used for this identity section.
raw=read(Path(R.B.S.env.DATA)/'train_X.csv');groups=collections.defaultdict(dict)
for r in raw:
    if r['row_id'][:3] in ['F13','F47']:groups[r['row_id'][:7]][int(r['row_id'][8:10])]=r
assert len(groups)==400
identity=json.loads((H/'input_identity_v1.json').read_text(encoding='utf-8'));assert identity['source_sha']==sha(H/'input_identity_v1.py') and identity['labels_read']==False and identity['new_fit']==0
valid=0;gap=0;h0_co2_observed=0;h0_co2_missing=0
for g in groups.values():
    assert 0 in g and 1 in g
    if g[0]['in_co2']=='':h0_co2_missing+=1
    else:h0_co2_observed+=1
    assert g[0]['act_heating']!='' and g[1]['act_heating']!=''
    a,b=float(g[0]['act_heating']),float(g[1]['act_heating']);prefix=(a+b)/2;gap=max(gap,abs(2*prefix-b-a));valid+=1
assert valid==400 and gap==0 and [c['rows'] for c in identity['checks']]==[400,400,400]
out=dict(status='PASS_TRACE_LINK_SUPPORT_ARITHMETIC_AND_INPUT_IDENTITY',new_fit=0,tree_rerouting=0,parent_tree_audit_review=dict(trees=600,nodes=parent['nodes'],node_mean_maxdiff=parent['node_mean_maxdiff'],ET_replay_maxdiff=parent['ET_replay_maxdiff'],ownership='root route reconstruction; critic source/receipt/hash review only'),support_profiles=profile_values,input_identity=dict(days=400,hour0_co2_observed=h0_co2_observed,hour0_co2_missing=h0_co2_missing,hour1_heating_reconstruction_maxdiff=gap,scope='Input identities only, not equal predictions or absent h0 information. This complete support verifier separately reads public OOF labels.'),retained45_XQ_exact_projection=True)
with (H/'critic_verify_support_identity_v2.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False),flush=True)

