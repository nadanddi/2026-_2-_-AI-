from pathlib import Path
import csv,json,sys,math,datetime,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
L=ROOT/'연구실/코덱스/local'/H.name
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def same(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10),(a,b)
def avg(v):return math.fsum(map(float,v))/len(v)
prep=json.loads((H/'preparation_v4.json').read_text(encoding='utf-8'));columns=prep['full_columns'];examples=read(H/'examples_v3.csv');groups={r['context'] for r in examples}
for context in groups:
    p=L/(context+'.npz');meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['npz_sha']
    with np.load(p,allow_pickle=False) as z:
        for r in [r for r in examples if r['context']==context]:
            role=r['role'];ids=z[role+'_row_id'].astype(str).tolist();i=ids.index(r['row_id']);same(r['ec'],z[role+'_y'][i])
            for j,c in enumerate(columns):same(r[c],z[role+'_X'][i,j])
            if role=='train':
                query='F47_161_00' if context=='actual1_7' else 'F13_112_00';qi=z['query_row_id'].astype(str).tolist().index(query);same(r['support_weight_for_query'],z['weights'][qi,i]);assert float(r['support_weight_for_query'])>0
            else:assert r['support_weight_for_query']==''
composition=json.loads((H/'composition_v1.json').read_text(encoding='utf-8'));p=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv';assert hashlib.sha256(p.read_bytes()).hexdigest()==composition['source_oof_sha'];oof=[r for r in read(p) if r['validator']=='DIAG10' and r['seed']=='7'];smooth=read(H/'summary_v1/smooth_day_support_profiles.csv');verified=[]
for result in composition['rows']:
    q=sorted([r for r in oof if r['farm']==result['farm'] and int(r['day'])==result['day']],key=lambda r:int(r['hour']));assert [int(r['hour']) for r in q]==list(range(24));assert len({r['fold'] for r in q})==1
    y=avg([r['y'] for r in q]);sm={}
    for name,column in [('ET','raw_et'),('LGB','raw_lgb'),('MLP','raw_mlp'),('PFN','old_pfn_raw')]:
        raw=[float(r[column]) for r in q];sm[name]=[.5*v+.5*avg(raw[:h+1]) for h,v in enumerate(raw)]
    weights=dict(ET=.48,LGB=.24,MLP=.08,PFN=.2);mix=[math.fsum(weights[n]*sm[n][h] for n in weights) for h in range(24)]
    for v,r in zip(mix,q):same(v,r['baseline']);assert float(r['clip_lo'])<=v<=float(r['clip_hi'])
    bias=avg(mix)-y;same(result['true_day'],y);same(result['final_prediction'],avg(mix));same(result['final_bias'],bias);same(result['ET_prediction'],avg(sm['ET']))
    contrib=[]
    for n,w in weights.items():v=w*(avg(sm[n])-y);same(result[n+'_weighted_bias'],v);contrib.append(v)
    same(math.fsum(contrib),bias);same(result['ET_fraction_of_signed_bias'],contrib[0]/bias);record=next(r for r in smooth if r['context']==result['context'] and r['farm']==result['farm'] and int(r['day'])==result['day']);same(record['prediction'],avg(sm['ET']));verified.append(dict(context=result['context'],farm=result['farm'],day=result['day'],signed_bias=bias,ET_signed_bias=contrib[0],ET_fraction_of_signed_bias=contrib[0]/bias))
out=dict(status='PASS_EXAMPLES_AND_FINAL_SIGNED_BIAS_LINK',example_rows=len(examples),feature_columns=len(columns),composition_rows=verified,clipping_active=False,new_fit=0,scope='seed7 actual OOF signed mean bias; not RMSE/SSE attribution or full3seed attribution; PFN support not traced')
path=H/('verify_trace_link_v2_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
