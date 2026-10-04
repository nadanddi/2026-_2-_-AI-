"""완성된 family21 공개 산출물 별도 CSV/Decimal/블록 합성 산술검산.
모델/runner/support/verifier import 없음. raw EC/test/EL1/fit/predict 없음.
"""
from pathlib import Path
import csv,json,hashlib,math,collections,decimal,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
SEEDS=(7,101,2024);ALPHA=.025/21
FOLDS=[(v,k) for v,n in [('DIAG10',10),('A',5),('B',5),('EXT10',1),('EXT12',1)] for k in range(n)]
COLS=['row_id','farm','day','hour','y','baseline','candidate','new_lgb_raw','raw_et','raw_lgb','raw_mlp','old_pfn_raw','validator','fold','seed','clip_lo','clip_hi']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def near(a,b,tol=1e-12):
    assert math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=tol,(a,b)
def rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:
        r=csv.DictReader(f);assert r.fieldnames==COLS
        return list(r)
def quantile(x,q):
    y=sorted(x);k=(len(y)-1)*q;i=math.floor(k);t=k-i
    return y[i] if i==len(y)-1 else y[i]*(1-t)+y[i+1]*t
def independent_boot(rs,seed):
    rec=collections.defaultdict(list)
    for r in rs:
        y,b,c=map(float,[r['y'],r['baseline'],r['candidate']])
        rec[r['farm'],int(r['day'])].append(((y-c)**2-(y-b)**2,int(r['hour'])))
    assert len(rs)==8640 and len(rec)==360
    for vals in rec.values():assert len(vals)==24 and {h for _,h in vals}==set(range(24))
    blocks={}
    for farm in ['F13','F47']:
        days=sorted(d for f,d in rec if f==farm)
        blocks[farm]=[(math.fsum(loss for day in days[i:i+5] for loss,_ in rec[farm,day]),sum(len(rec[farm,day]) for day in days[i:i+5])) for i in range(0,len(days),5)]
    rng=np.random.default_rng(20261003+seed)
    draws={f:rng.integers(len(blocks[f]),size=(20000,len(blocks[f]))) for f in ['F13','F47']}
    sample=[]
    for i in range(20000):
        totals=[];count=0
        for f in ['F13','F47']:
            for j in draws[f][i]:
                loss,n=blocks[f][j];totals.append(loss);count+=n
        sample.append(math.fsum(totals)/count)
    return dict(p_worse=sum(x>=0 for x in sample)/20000,ci_adjusted=[quantile(sample,q) for q in [ALPHA,1-ALPHA]],ci95=[quantile(sample,q) for q in [.025,.975]],draws=20000,blocks={f:len(b) for f,b in blocks.items()})
def main():
    report=readj(H/'full_verification_v3.json')
    assert report['status']=='PASS' and report['cells']==66 and report['rows']==83160 and report['score_cells']==15 and report['family']==21
    assert report['alpha']==ALPHA and report['verification_fit']==report['verification_predict']==0
    for name,file in [('verifier_sha256','verify_full_v3.py'),('runner_sha256','run_v3.py'),('preparation_sha256','preparation_v3.json'),('preregistration_sha256','preregistration_v1.md'),('fit_audit_sha256','fit_audit_v1.json'),('first_audit_sha256','first_fold_verification_v1.json'),('scores_sha256','full_scores_v3.csv'),('segments_sha256','full_segments_v3.csv')]:assert report[name]==sha(H/file)
    assert report['aggregate_sha256']==sha(OUT/'oof.csv')
    prep=readj(H/'preparation_v3.json');fit=readj(H/'fit_audit_v1.json');first=readj(H/'first_fold_verification_v1.json')
    assert fit['preparation']==prep and fit['score_count']==0 and prep['fit_count']==prep['predict_count']==prep['score_count']==0
    assert prep['manifest']==report['manifest'] and prep['original_r3_guard']==report['r3_cache_audit']
    assert len(prep['manifest'])==22 and [(m['validator'],m['fold']) for m in prep['manifest']]==FOLDS
    assert report['runtime']==prep['runtime'] and report['dependencies']==prep['dependencies'] and report['inputs']==prep['inputs']
    for rel,digest in prep['manifest'][0]['cache_hashes'].items():assert sha(ROOT/rel)==digest
    for m in prep['manifest']:
        assert len(m['full_features'])==38 and len(m['old_lgb_features'])==14 and len(m['new_lgb_features'])==23
        assert m['new_lgb_features'][:14]==m['old_lgb_features'] and m['old_lgb_features'][-1]=='season' and 'day' not in m['new_lgb_features']
        assert m['dependencies']==prep['dependencies'] and m['runtime']==prep['runtime'] and m['inputs']==prep['inputs']
        for rel,digest in m['cache_hashes'].items():assert sha(ROOT/rel)==digest
    assert first['status']=='PASS' and first['signature']==fit['cells'][0]['signature'] and first['atol']==1e-12
    assert set(first['raw_errors'])=={'repeat','fresh_fit','single','reversed','other_query'}
    for x in list(first['raw_errors'].values())+[first['original_lgb_maxdiff'],first['scalar_maxdiff']]:assert math.isfinite(x) and 0<=x<=1e-12
    metas=[];conc=[];groups=collections.defaultdict(list);prefix_max=0.
    for i,(v,k) in enumerate(FOLDS):
        for s in SEEDS:
            p=OUT/f'{v}_{k}_{s}_pred.csv';meta=readj(OUT/f'{v}_{k}_{s}.json');metas.append(meta)
            sig=dict(**prep['manifest'][i],seed=s,preparation_sha256=report['preparation_sha256'],preregistration_sha256=report['preregistration_sha256'])
            assert meta['signature']==sig and meta['csv_sha256']==sha(p) and meta['status']=='PASS'
            if (v,k,s)==('DIAG10',0,7):assert meta['first_audit_sha256']==report['first_audit_sha256']
            else:assert 'first_audit_sha256' not in meta
            rs=rows(p);assert len({r['row_id'] for r in rs})==len(rs)
            ids='\n'.join(r['row_id'] for r in rs);assert hashlib.sha256(ids.encode()).hexdigest()==sig['query_ids']
            assert all(r['validator']==v and int(r['fold'])==k and int(r['seed'])==s for r in rs)
            for key in ['day','hour','seed','fold']:assert all(str(int(r[key]))==r[key] for r in rs)
            for r in rs:
                assert all(math.isfinite(float(r[c])) for c in COLS if c not in ['row_id','farm','validator'])
                near(float(r['clip_lo']),sig['bounds'][0]);near(float(r['clip_hi']),sig['bounds'][1])
            histories={name:collections.defaultdict(list) for name in ['baseline','candidate']}
            for r in sorted(rs,key=lambda x:(x['farm'],int(x['day']),int(x['hour']))):
                for name,lgb in [('baseline','raw_lgb'),('candidate','new_lgb_raw')]:
                    raw=math.fsum([.48*float(r['raw_et']),.24*float(r[lgb]),.08*float(r['raw_mlp']),.2*float(r['old_pfn_raw'])])
                    hist=histories[name][r['farm'],int(r['day'])];hist.append(raw)
                    result=min(float(r['clip_hi']),max(float(r['clip_lo']),.5*raw+.5*math.fsum(hist)/len(hist)))
                    near(result,float(r[name]));prefix_max=max(prefix_max,abs(result-float(r[name])))
            conc.extend(rs);groups[v,s].extend(rs)
    assert metas==fit['cells'] and len(conc)==83160 and len({(r['validator'],r['fold'],r['seed'],r['row_id']) for r in conc})==83160
    assert conc==rows(OUT/'oof.csv'),'CSV text roundtrip aggregate equality'
    with (H/'full_scores_v3.csv').open(encoding='utf-8',newline='') as f:score_rows=list(csv.DictReader(f))
    assert len(score_rows)==15 and {(r['validator'],int(r['seed'])) for r in score_rows}==set(groups)
    with (H/'full_segments_v3.csv').open(encoding='utf-8',newline='') as f:segment_rows=list(csv.DictReader(f))
    context=decimal.Context(prec=45);D=context.create_decimal;metrics=[];boot_results={}
    def rmse(rs,name):
        terms=[context.multiply(context.subtract(D(r['y']),D(r[name])),context.subtract(D(r['y']),D(r[name]))) for r in rs]
        total=sum(terms,decimal.Decimal(0));return float(context.sqrt(context.divide(total,D(len(rs)))))
    for r in score_rows:
        v,s=r['validator'],int(r['seed']);rs=groups[v,s];assert len(rs)==int(r['n'])
        rb,rc=rmse(rs,'baseline'),rmse(rs,'candidate')
        near(rb,r['baseline']);near(rc,r['candidate']);near(rc-rb,r['delta_rmse']);near(100*(rc/rb-1),r['change_pct'])
        assert (rc<rb)==(float(r['candidate_numpy'])<float(r['baseline_numpy']))
        metrics.append(dict(validator=v,seed=s,n=len(rs),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1)))
    for s in SEEDS:
        rs=groups['DIAG10',s];day_y=collections.defaultdict(list)
        for r in rs:day_y[r['farm'],int(r['day'])].append(float(r['y']))
        high={k for k,ys in day_y.items() if math.fsum(ys)/len(ys)>=1};assert len(high)==31 and len(day_y)==360
        b=independent_boot(rs,s);ref=report['bootstrap'][str(s)];near(b['p_worse'],ref['p_worse'],0.)
        for name in ['ci_adjusted','ci95']:
            for a,c in zip(b[name],ref[name]):near(a,c)
        assert (b['p_worse']<ALPHA and b['ci_adjusted'][1]<0)==(ref['p_worse']<ALPHA and ref['ci_adjusted'][1]<0)
        boot_results[s]=b
        subsets={'high':[r for r in rs if (r['farm'],int(r['day'])) in high], 'ordinary':[r for r in rs if (r['farm'],int(r['day'])) not in high], 'late':[r for r in rs if int(r['day'])>=179], 'F13':[r for r in rs if r['farm']=='F13'], 'F47':[r for r in rs if r['farm']=='F47'], 'hour0':[r for r in rs if int(r['hour'])==0]}
        assert len(subsets['high'])==744 and len(subsets['ordinary'])==7896
        for name,g in subsets.items():
            matches=[r for r in segment_rows if int(r['seed'])==s and r['segment']==name];assert len(matches)==1;ref=matches[0]
            assert int(ref['n'])==len(g) and int(ref['days'])==len({(r['farm'],r['day']) for r in g})
            for pred in ['baseline','candidate']:
                near(rmse(g,pred),ref[pred]);near(math.fsum(float(r[pred])-float(r['y']) for r in g)/len(g),ref[pred+'_bias'])
    passed=all(m['candidate']<m['baseline'] for m in metrics) and all(b['p_worse']<ALPHA and b['ci_adjusted'][1]<0 for b in boot_results.values())
    assert passed==report['public_pass']
    result=dict(status='PASS_INDEPENDENT_COMPLETE_PUBLIC_CROSSCHECK',public_pass=passed,parent_report_sha256=sha(H/'full_verification_v3.json'),crosschecker_sha256=sha(Path(__file__)),rows=83160,cells=66,metrics=metrics,bootstrap=boot_results,high_days=31,ordinary_days=329,prefix_maxdiff=prefix_max,method=['stdlib csv exact aggregate text equality','Decimal45 squared loss/RMSE','scalar fsum clipped prefix mixing','row->farm/day->five-day scalar block totals','20k same pinned RNG but scalar draw sums/manual sorted quantiles'],fit=0,predict=0,test_reads=0,raw_ec_reads=0,EL1_rescore=0,limitations=['public repeated validation','saved audits checked; no fit/predict replay','RNG intentionally identical for reproducibility; independent statistic and quantile arithmetic'])
    with (H/'crosscheck_complete_result_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('PASS_INDEPENDENT_COMPLETE_PUBLIC_CROSSCHECK',passed,prefix_max)
if __name__=='__main__':main()
