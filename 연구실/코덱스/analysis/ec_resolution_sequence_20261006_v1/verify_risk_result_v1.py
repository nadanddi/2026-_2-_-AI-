"""Read-only trained-parameter reconstruction + independent final screen arithmetic."""
from pathlib import Path
import sys,csv,json,math,hashlib,datetime
from collections import defaultdict
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(H));import verify_stage1_v1 as B
np,pd=B.np,B.pd
from risk_core_v2 import design
L=ROOT/'연구실/코덱스/local'/H.name;O=L/'risk_stage2_v1';D=H/'stage34_results_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as s:return list(csv.DictReader(s))
def mean(v):return math.fsum(map(float,v))/len(v)
def same(a,b,tol=1e-10):assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)
def sigmoid(x):return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
prep=js(H/'stage2_preparation_v1.json');rec=js(H/'stage2_receipt_v1.json');result=js(D/'completion.json');cfg=prep['config'];assert not (O/'worker.lock').exists();assert sha(H/'stage2_v1.py')==prep['source_sha']==rec['source_sha'];assert sha(H/'risk_core_v2.py')==prep['helper_sha'];assert sha(H/'PROTOCOL2_v1.md')==prep['protocol_sha']==result['protocol_sha'];assert sha(H/'stage2_preparation_v1.json')==rec['prep_sha'];assert sha(O/'risk_rows.csv')==rec['output_sha'];assert sha(H/'stage2_receipt_v1.json')==result['stage2_receipt_sha'];assert sha(H/'stage3_4_v2.py')==result['source_sha'];assert sha(L/'stage34_rows_v2.csv')==result['row_output_sha']
riskrows=read(O/'risk_rows.csv');riskindex={(int(r['k']),int(r['seed']),r['row_id']):r for r in riskrows};assert len(riskrows)==len(riskindex)==10152
fitchecks=[];maxprob=0
for item in prep['inputs']:
    k,seed=item['k'],item['seed'];tp=L/'risk_inputs_v1'/f'train_{k}_{seed}.csv';qp=L/'risk_inputs_v1'/f'query_{k}_{seed}.csv';assert sha(tp)==item['train_sha'] and sha(qp)==item['query_sha'];tr=pd.read_csv(tp,float_precision='round_trip');q=pd.read_csv(qp,float_precision='round_trip');x=design(tr,prep['base_whitelist']);z=design(q,prep['base_whitelist']);assert list(x)==list(z)==prep['feature_columns'];assert not set(tr.row_id)&set(q.row_id)
    meta=js(O/f'fit_{k}_{seed}.json');assert meta['train_sha']==item['train_sha'] and meta['query_sha']==item['query_sha'] and meta['columns']==list(x)
    y=(tr.A.to_numpy()-tr.sub_ec.to_numpy()>cfg['label_threshold']).astype(float);assert int(y.sum())==meta['positives'];counts=tr.groupby(['farm','day']).A.transform('size').to_numpy();assert np.all(counts==24);weights=np.array([1/24]*len(tr));same(math.fsum(weights),meta['sum_weight']);assert meta['train_rows']==len(tr) and meta['query_rows']==len(q)
    if meta['constant']:
        assert len(set(y))==1;prob=[mean(y)]*len(q);grad=None
    else:
        assert meta['classes']==[0,1] and meta['iterations']<cfg['max_iter'];xt=x.to_numpy(float);zt=z.to_numpy(float);med=np.nanmedian(xt,axis=0);med=np.where(np.isnan(med),0,med);np.testing.assert_allclose(med,meta['median'],atol=1e-12,rtol=1e-12);xt=np.where(np.isnan(xt),med,xt);zt=np.where(np.isnan(zt),med,zt)
        mu=xt.mean(axis=0);scale=np.sqrt(((xt-mu)**2).mean(axis=0));scale=np.where(scale==0,1,scale);np.testing.assert_allclose(mu,meta['mean'],atol=1e-10,rtol=1e-12);np.testing.assert_allclose(scale,meta['scale'],atol=1e-10,rtol=1e-12)
        # Use independently reconstructed transformations and recorded fitted coefficients.
        coef=np.asarray(meta['coef']);inter=meta['intercept'];st=(xt-mu)/scale;sq=(zt-mu)/scale;prob=[sigmoid(math.fsum(float(a)*float(b) for a,b in zip(row,coef))+inter) for row in sq]
        ptrain=np.array([sigmoid(math.fsum(float(a)*float(b) for a,b in zip(row,coef))+inter) for row in st]);residual=weights*(ptrain-y);gcoef=(st.T@residual+coef/cfg['C'])/math.fsum(weights);gi=math.fsum(residual)/math.fsum(weights);grad=max(float(np.max(abs(gcoef))),abs(gi));assert grad<1e-6,('unexpected nonstationary LR optimum',k,seed,grad)
    for i,row in enumerate(q.itertuples()):
        r=riskindex[k,seed,row.row_id];err=abs(prob[i]-float(r['risk']));maxprob=max(maxprob,err);same(prob[i],r['risk'],1e-11)
        for col in ['A','sub_ec','smooth_lgb','act_vent_tdz','act_circfan_tdm']:same(getattr(row,col),r[col])
        assert row.farm==r['farm'] and int(row.day)==int(r['day']) and int(row.hour)==int(r['hour'])
    fitchecks.append(dict(k=k,seed=seed,rows=len(tr),positives=int(y.sum()),weight_sum=math.fsum(weights),iterations=meta.get('iterations'),normalized_L2_gradient_max=grad))
rows=read(L/'stage34_rows_v2.csv');assert len(rows)==10152;group=defaultdict(list)
for r in rows:
    for c in ['A','sub_ec','smooth_lgb','act_vent_tdz','act_circfan_tdm','risk','candidate','delta','y_day']:r[c]=float(r[c])
    for c in ['seed','k','day','hour']:r[c]=int(r[c])
    raw=riskindex[r['k'],r['seed'],r['row_id']]
    for c in ['A','sub_ec','smooth_lgb','act_vent_tdz','act_circfan_tdm','risk']:same(r[c],raw[c])
    eligible=r['act_vent_tdz']>=cfg['vent_zero'] and r['act_circfan_tdm']<cfg['fan_mean'];assert eligible==(r['eligible']=='True')
    amplitude=min(max(r['A']-r['smooth_lgb'],0),cfg['cap'],r['A']);delta=amplitude*max(0,min(1,(r['risk']-cfg['threshold'])/(1-cfg['threshold']))) if eligible else 0;candidate=r['A']-delta;same(candidate,r['candidate']);same(r['A']-candidate,r['delta']);assert 0<=candidate<=r['A'] and delta<=cfg['cap']+1e-15;group[r['seed'],r['farm'],r['day']].append(r)
for key,q in group.items():
    assert len(q)==24 and {r['hour'] for r in q}==set(range(24));ym=mean([r['sub_ec'] for r in q])
    for r in q:same(r['y_day'],ym);assert (ym>=1)==(r['high']=='True')
def metric(q):
    if not q:return dict(rows=0,days=0,baseline=None,candidate=None,change_pct=None,mse_delta=None)
    ae=[r['A']-r['sub_ec'] for r in q];be=[r['candidate']-r['sub_ec'] for r in q];a=math.sqrt(mean([e*e for e in ae]));b=math.sqrt(mean([e*e for e in be]));return dict(rows=len(q),days=len({(r['farm'],r['day']) for r in q}),baseline=a,candidate=b,change_pct=(b/a-1)*100,mse_delta=mean([v*v-u*u for u,v in zip(ae,be)]))
def compare(a,b):
    for c,v in a.items():
        if v is None:assert b[c] is None
        else:same(v,b[c])
def auc(q):
    items=sorted((r['risk'],r['A']-r['sub_ec']>cfg['label_threshold']) for r in q);positives=sum(t for _,t in items);neg=len(items)-positives
    if not positives or not neg:return None
    ranktotal=0.;i=0
    while i<len(items):
        j=i+1
        while j<len(items) and items[j][0]==items[i][0]:j+=1
        ranktotal+=((i+1+j)/2)*sum(t for _,t in items[i:j]);i=j
    return (ranktotal-positives*(positives+1)/2)/(positives*neg)
reasons=[];stats={};riskchecks=[]
for seed in [7,101,2024]:
    q=[r for r in rows if r['seed']==seed];assert len(q)==3384;subsets={'all':q,'ordinary':[r for r in q if r['high']=='False'],'high':[r for r in q if r['high']=='True'],'pass1':[r for r in q if r['day']<179],'pass2':[r for r in q if r['day']>=179]}
    subsets.update({'farm_'+f:[r for r in q if r['farm']==f] for f in ['F13','F47']});subsets.update({'fold_'+str(k):[r for r in q if r['k']==k] for k in range(4)})
    for name,qq in subsets.items():m=metric(qq);stats[seed,name]=m;official=next(r for r in result['seed_scores'] if r['seed']==seed and r['subset']==name);compare(m,official)
    for name in ['all','ordinary']:
        m=stats[seed,name]
        if not m['rows'] or not m['candidate']<m['baseline']:reasons.append(f'{seed}:{name}:no_strict_improvement')
    m=stats[seed,'high']
    if not m['rows'] or m['candidate']>m['baseline']+1e-12:reasons.append(f'{seed}:high:protection_failed')
    m=stats[seed,'pass2']
    if m['rows'] and m['change_pct']>=2:reasons.append(f'{seed}:pass2:guard_failed')
    official=next(r for r in result['risk_metrics'] if r['seed']==seed);target=[r['A']-r['sub_ec']>cfg['label_threshold'] for r in q];same(sum(target),official['positive_rows']);value=auc(q);same(value,official['auc']);same(sum(r['eligible']=='True' for r in q),official['eligible_rows']);same(sum(r['delta']>0 and r['high']=='True' for r in q),official['modified_high_rows'])
    for name,selected in [('risk_flag',[r['risk']>=cfg['threshold'] for r in q]),('modified',[r['delta']>0 for r in q])]:
        n=sum(selected);tp=sum(t and s for t,s in zip(target,selected));compare(dict(selected=n,true_positive=tp,precision=tp/n if n else None,recall=tp/sum(target) if sum(target) else None),official[name])
    riskchecks.append(dict(seed=seed,auc=value,high_modified_rows=official['modified_high_rows'],pass2_tested=False))
assert reasons==result['reasons']==['101:high:protection_failed'] and result['status']=='SCREEN_REJECT'
for r in read(D/'daily_scores.csv'):
    q=group[int(r['seed']),r['farm'],int(r['day'])];m=metric(q)
    for c,v in m.items():same(v,r[c])
    same(mean([u['sub_ec'] for u in q]),r['ymean']);same(mean([u['A'] for u in q]),r['pmean']);same(mean([u['candidate'] for u in q]),r['cmean']);same(sum(u['delta']>0 for u in q),r['modified_rows']);same(max(u['risk'] for u in q),r['risk_max'])
byrow=defaultdict(list)
for r in rows:byrow[r['row_id']].append(r)
av=[]
for rid,q in byrow.items():assert {r['seed'] for r in q}=={7,101,2024};r=q[0].copy();r['A']=mean([u['A'] for u in q]);r['candidate']=mean([u['candidate'] for u in q]);av.append(r)
for name,q in [('all',av),('ordinary',[r for r in av if r['high']=='False']),('high',[r for r in av if r['high']=='True'])]:compare(metric(q),result['seed_mean'][name])
modified=[r for r in rows if r['delta']>0];assert len(modified)==result['modified_rows']==57;assert len({(r['farm'],r['day']) for r in modified})==result['modified_unique_days']==4
highmods=[dict(row_id=r['row_id'],seed=r['seed'],y=r['sub_ec'],A=r['A'],candidate=r['candidate'],risk=r['risk'],delta=r['delta'],sse_change=(r['candidate']-r['sub_ec'])**2-(r['A']-r['sub_ec'])**2) for r in modified if r['high']=='True'];assert len(highmods)==1 and highmods[0]['sse_change']>0
out=dict(status='PASS_AUDIT_SCREEN_REJECT',candidate_status='SCREEN_REJECT',reasons=reasons,fit_cells=len(fitchecks),audit_new_fit=0,fit_parameter_checks=fitchecks,max_risk_reconstruction_error=maxprob,rows=10152,unique_days=141,ordinary_days=133,high_days=8,modified_seed_rows=57,modified_unique_days=4,modified_days=sorted({r['farm']+'_'+str(r['day']) for r in modified}),high_modified_rows=highmods,risk_checks=riskchecks,seed_mean=result['seed_mean'],pass2_status='UNTESTED_ZERO_ROWS_ALL_SEEDS',full80=False,adoption=False,limitations=['audit pass is arithmetic/provenance verification, not efficacy pass','three seeds share dates;57 modifications are seed-row events','gradient threshold1e-6 is audit stationarity check, not a changed candidate acceptance rule'])
path=H/('verify_risk_result_v1_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(status=out['status'],fit_cells=len(fitchecks),max_risk_error=maxprob,max_normalized_gradient=max(r['normalized_L2_gradient_max'] for r in fitchecks),reasons=reasons,modified_rows=57,highmods=highmods,output=str(path)),ensure_ascii=False,indent=2))
