"""Independent replay of completed public caches. Never fit or select weights."""
from pathlib import Path
import sys,csv,json,math,hashlib,statistics
from collections import defaultdict
from decimal import Decimal,localcontext
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
AN=ROOT/'집/코덱스/analysis';LOCAL=ROOT/'집/코덱스/local'
H=AN/'ec_nested_log_blend_20261004_v1';SRC=LOCAL/H.name
INNER=LOCAL/'ec_matched_inner_calibration_20261003_v1'
LOG=LOCAL/'ec_log_partition_mean_20261004_v1'
PUBLIC=LOCAL/'ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
OUT=Path(__file__).resolve().parent
def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def idhash(a):return hashlib.sha256('\n'.join(str(x) for x in a).encode()).hexdigest()
def avg(a):
    a=list(a);return math.fsum(a)/len(a)
gaps=defaultdict(float)
def compare(a,b,name):
    a=np.asarray(a,float);b=np.asarray(b,float)
    assert a.shape==b.shape and a.size>0
    assert np.isfinite(a).all() and np.isfinite(b).all()
    gap=float(np.max(np.abs(a-b)));gaps[name]=max(gaps[name],gap)
    assert gap<1e-12,(name,gap)
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def parts(row_id):return row_id[:3],int(row_id[4:7]),int(row_id[8:10])
def shrink(values,ids):
    groups=defaultdict(list);result=np.empty(len(ids))
    for i,row in enumerate(ids):
        f,d,h=parts(str(row));groups[f,d].append((h,i,float(values[i])))
    for rr in groups.values():
        history=[]
        for h,i,value in sorted(rr):
            history.append(value);result[i]=.5*value+.5*avg(history)
    return result
def seasonal_baseline(ids,y,season,qids,qseason):
    days=defaultdict(list)
    for row,value,s in zip(ids,y,season):
        f,d,h=parts(str(row));days[f,d].append((float(value),float(s)))
    maps={}
    for farm in ['F13','F47']:
        keys=sorted(k for k in days if k[0]==farm)
        seas=np.array([days[k][0][1] for k in keys]);labels=np.array([avg(v for v,s in days[k]) for k in keys])
        assert all(max(s for v,s in days[k])-min(s for v,s in days[k])<1e-12 for k in keys)
        order=np.argsort(seas,kind='quicksort');seas=seas[order];labels=labels[order]
        med=[]
        for i in range(len(labels)):
            window=labels[max(0,i-10):min(len(labels),i+11)]
            med.append(float(statistics.median(window)) if len(window)>=5 else None)
        good=[i for i,z in enumerate(med) if z is not None];assert good
        for i,z in enumerate(med):
            if z is None:med[i]=med[next((j for j in good if j>i),good[-1])]
        maps[farm]=seas,np.array(med)
    return np.array([np.interp(s,*maps[parts(str(row))[0]]) for row,s in zip(qids,qseason)])

report=json.loads((H/'full_verification_v1.json').read_text(encoding='utf-8'))
fit=json.loads((H/'fit_audit_v1.json').read_text(encoding='utf-8'));assert fit['status']=='PASS' and len(fit['cells'])==66
source=sha(H/'run_v3.py');assert source==report['source_sha256']==(SRC/'source_sha.txt').read_text()
assert source=='f3a66402c3dba9fff5585d9c64e478d2f271fb57593c2f683682d72db0d9f542'
scores={(r['validator'],int(r['seed'])):r for r in table(H/'full_scores_v1.csv')}
segments={(int(r['seed']),r['segment']):r for r in table(H/'full_segments_v1.csv')}
public=table(PUBLIC);publichash=sha(PUBLIC)
truth={r['row_id']:float(r['sub_ec']) for r in public if r['validator']=='DIAG10' and int(r['seed'])==7}
assert len(truth)==8640
outer_ref={(r['validator'],int(r['validation_fold']),int(r['seed']),r['row_id']):r for r in public}
groups=defaultdict(list);seen=set();cells=[];csvhash={};npzhash={}
expected_runtime=dict(python='3.12.14',numpy='2.5.3',pandas='3.0.1',sklearn='1.9.1',lightgbm='4.7.0')
for split in report['split_audit']:
    v,k=split['validator'],int(split['fold']);assert v in ['DIAG10','A','B','EXT10','EXT12']
    cpu_path=INNER/f'{v}_{k}_cpu.npz';cpu=load(cpu_path)
    pfnpaths=[INNER/f'{v}_{k}_pfn_{s}.npz' for s in [1,2,3,4]]
    pfns=[load(p) for p in pfnpaths]
    for z in pfns:assert np.array_equal(z['row_id'],cpu['row_id'])
    bag=np.mean([z['prediction'] for z in pfns],axis=0)
    for seed in [7,101,2024]:
        stem=f'{v}_{k}_{seed}';path=SRC/f'{stem}.csv';npz=SRC/f'{stem}_inner.npz'
        rows=table(path);meta=json.loads((SRC/f'{stem}.json').read_text(encoding='utf-8'));z=load(npz)
        sig=meta['signature'];assert meta['status']=='PASS' and sig['run_source_sha256']==source
        assert (sig['validator'],sig['fold'],sig['seed'])==(v,k,seed) and sig['runtime']==expected_runtime
        csvhash[stem]=sha(path);npzhash[stem]=sha(npz)
        assert csvhash[stem]==meta['output_sha256'] and npzhash[stem]==meta['inner_arrays_sha256']
        assert sig['public_cache_sha256']==publichash and sig['cpu_sha256']==sha(cpu_path)
        assert sig['pfn_sha256']==[sha(p) for p in pfnpaths]
        assert sig['log_cache_sha256']==sha(LOG/f'{stem}.csv')
        assert np.array_equal(z['row_id'],cpu['row_id']) and np.array_equal(z['train_row_id'],cpu['inner_train_id'])
        for key,array in [('inner_train_ids',z['train_row_id']),('inner_query_ids',z['row_id']),('inner_targets',z['train_y']),('inner_query_targets',z['y'])]:
            assert sig[key]==idhash(array)
        assert len(set(z['row_id']))==len(z['row_id']) and len(set(z['train_row_id']))==len(z['train_row_id'])
        assert set(z['row_id']).isdisjoint(z['train_row_id'])
        banned={(f,d+offset) for row in z['row_id'] for f,d,h in [parts(str(row))] for offset in [-1,0,1]}
        assert not {(f,d) for row in z['train_row_id'] for f,d,h in [parts(str(row))]}&banned
        compare(z['y'],[truth[str(row)] for row in z['row_id']],'query_labels')
        compare(z['train_y'],[truth[str(row)] for row in z['train_row_id']],'train_labels')
        lo,hi=float(z['inner_lo']),float(z['inner_hi'])
        assert lo==float(np.min(z['train_y'])) and hi==float(np.max(z['train_y']))
        assert meta['inner_bounds']==[lo,hi] and float(cpu['lo'])==lo and float(cpu['hi'])==hi
        for a in ['b_train','b_query','ratio_train']:assert np.isfinite(z[a]).all() and (z[a]>0).all()
        compare(z['ratio_train'],z['train_y']/z['b_train'],'ratio_definition')
        compare(z['b_train'],seasonal_baseline(z['train_row_id'],z['train_y'],z['train_season'],z['train_row_id'],z['train_season']),'rolling_b_train')
        compare(z['b_query'],seasonal_baseline(z['train_row_id'],z['train_y'],z['train_season'],z['row_id'],z['query_season']),'rolling_b_query')
        ib=np.clip(shrink(.8*cpu[f'r3_{seed}']+.2*bag,z['row_id']),lo,hi)
        old=shrink(z['old_et_raw'],z['row_id']);new=shrink(z['log_raw'],z['row_id'])
        compare(z['baseline'],ib,'matched_inner_baseline');compare(z['old_et_shrunk'],old,'old_shrink');compare(z['log_shrunk'],new,'log_shrink')
        proposal=ib+.48*(new-old);endpoint=np.clip(proposal,lo,hi)
        compare(z['log_endpoint'],endpoint,'inner_endpoint')
        e=ib-z['y'];direction=endpoint-ib
        compare(z['e'],e,'inner_e');compare(z['direction'],direction,'inner_direction')
        num=avg(float(x)*float(y) for x,y in zip(e,direction));d2=avg(float(x)**2 for x in direction);den=d2+.01
        w=min(1,max(0,-num/den));grad=2*(num+den*w)
        assert (w==0 and grad>=-1e-12) or (w==1 and grad<=1e-12) or abs(grad)<1e-12
        baseline_mse=avg(float(x)**2 for x in e);fitted_mse=avg(float(x+w*y)**2 for x,y in zip(e,direction))
        close_mse=baseline_mse+2*num*w+d2*w*w;compare([fitted_mse],[close_mse],'quadratic_identity')
        assert fitted_mse+.01*w*w<=baseline_mse+1e-12
        for col,value in dict(n=len(e),numerator=num,direction_mse=d2,denominator=den,weight=w,gradient=grad,baseline_mse=baseline_mse,fitted_mse=fitted_mse,penalized_fitted_loss=fitted_mse+.01*w*w).items():
            compare([value],[meta['weight_fit'][col]],'weight_sufficient_statistics')
        with localcontext() as context:
            context.prec=45
            decimal_num=sum((Decimal(float(x))*Decimal(float(y)) for x,y in zip(e,direction)),Decimal(0))/Decimal(len(e))
            decimal_d2=sum((Decimal(float(x))**2 for x in direction),Decimal(0))/Decimal(len(e))
            decimal_w=float(max(Decimal(0),min(Decimal(1),-decimal_num/(decimal_d2+Decimal('.01')))))
            compare([w],[decimal_w],'decimal_weight')
        prior=table(LOG/f'{stem}.csv');assert [r['row_id'] for r in rows]==[r['row_id'] for r in prior]
        assert sig['outer_query_ids']==idhash([r['row_id'] for r in rows])
        assert sig['outer_targets']==idhash(np.asarray([float(r['y']) for r in rows]))
        out_lo,out_hi=sig['bounds'];clip_count=0
        for r,oldr in zip(rows,prior):
            key=v,k,seed,r['row_id'];assert key not in seen;seen.add(key)
            assert r['validator']==v and int(r['fold'])==k and int(r['seed'])==seed
            ref=outer_ref[key]
            for name in ['farm','day','hour']:assert r[name]==oldr[name] and str(r[name])==str(ref[name])
            r.update(yy=float(r['y']),bb=float(r['baseline']),cc=float(r['candidate']),dd=float(r['direction']),ll=float(r['log_endpoint']))
            compare([r['yy'],r['bb'],r['ll']],[float(oldr['y']),float(oldr['baseline']),float(oldr['candidate'])],'outer_log_identity')
            compare([r['yy'],r['bb']],[float(ref['sub_ec']),float(ref['season_v2'])],'actual_v2_identity')
            compare([r['dd'],float(r['weight'])],[r['ll']-r['bb'],w],'outer_direction_weight')
            final=r['bb']+w*r['dd'];compare([r['cc']],[final],'outer_scalar')
            assert out_lo<=r['bb']<=out_hi and out_lo<=r['ll']<=out_hi and out_lo<=r['cc']<=out_hi
            clip_count+=not out_lo<=final<=out_hi
        assert clip_count==meta['outer_mix_clipped_rows']==0
        inner_rec=next(a for a in report['inner_audit'] if (a['validator'],a['fold'],a['seed'])==(v,k,seed))
        compare([w,grad,len(e),baseline_mse,fitted_mse],[inner_rec[c] for c in ['weight','gradient','n_query','inner_baseline_mse','inner_fitted_mse']],'saved_inner_audit')
        out_e=[r['bb']-r['yy'] for r in rows];out_d=[r['dd'] for r in rows]
        out_num=avg(a*b for a,b in zip(out_e,out_d));out_d2=avg(a*a for a in out_d)
        delta_sse=math.fsum((r['cc']-r['yy'])**2-(r['bb']-r['yy'])**2 for r in rows)
        compare([delta_sse/len(rows)],[2*w*out_num+w*w*out_d2],'outer_quadratic_identity')
        cells.append(dict(validator=v,fold=k,seed=seed,n_inner=len(e),n_outer=len(rows),weight=w,inner_num=num,inner_d2=d2,inner_den=den,gradient=grad,
                          inner_baseline_mse=baseline_mse,inner_fitted_mse=fitted_mse,
                          outer_direction_cross=out_num,outer_d2=out_d2,outer_delta_sse=delta_sse))
        groups[v,seed].extend(rows)
assert len(seen)==83160 and len(cells)==len(csvhash)==len(npzhash)==66

def metrics(rr):
    out={}
    for alias,col in [('baseline','bb'),('candidate','cc')]:
        e=[r[col]-r['yy'] for r in rr];sse=math.fsum(x*x for x in e)
        out[alias]=math.sqrt(sse/len(rr));out[alias+'_bias']=avg(e);out[alias+'_sse']=sse
    out['change_pct']=100*(out['candidate']/out['baseline']-1);out['delta_sse']=out['candidate_sse']-out['baseline_sse']
    return out
scorecheck=[];segmentcheck=[];bootstrap=[]
for (v,seed),rr in sorted(groups.items()):
    actual=metrics(rr);saved=scores[v,seed];assert len(rr)==int(saved['n'])
    for col in ['baseline','candidate','change_pct']:compare([actual[col]],[float(saved[col])],'score_'+col)
    with localcontext() as context:
        context.prec=45
        for alias,col in [('baseline','bb'),('candidate','cc')]:
            sse=sum(((Decimal(r[col])-Decimal(r['yy']))**2 for r in rr),Decimal(0))
            compare([actual[alias]],[float((sse/Decimal(len(rr))).sqrt())],'decimal_score')
    scorecheck.append(dict(validator=v,seed=seed,n=len(rr),**actual))
for seed in [7,101,2024]:
    rr=groups['DIAG10',seed];assert len(rr)==len({r['row_id'] for r in rr})==8640
    days=defaultdict(list)
    for r in rr:days[r['farm'],int(r['day'])].append(r)
    assert len(days)==360 and all(len(r)==24 for r in days.values())
    high={k for k,r in days.items() if avg(z['yy'] for z in r)>=1};assert len(high)==31
    masks=dict(high=lambda r:(r['farm'],int(r['day'])) in high,ordinary=lambda r:(r['farm'],int(r['day'])) not in high,
               late=lambda r:int(r['day'])>=179,F13=lambda r:r['farm']=='F13',F47=lambda r:r['farm']=='F47',hour0=lambda r:int(r['hour'])==0)
    for name,mask in masks.items():
        sub=[r for r in rr if mask(r)];actual=metrics(sub);saved=segments[seed,name]
        assert len(sub)==int(saved['n']) and len({(r['farm'],r['day']) for r in sub})==int(saved['days'])
        for col in ['baseline','candidate','baseline_bias','candidate_bias']:compare([actual[col]],[float(saved[col])],'segment_'+col)
        segmentcheck.append(dict(seed=seed,segment=name,n=len(sub),**actual))
    rng=np.random.default_rng(20261003+seed);ss=np.zeros(20000);nn=np.zeros(20000)
    for farm in ['F13','F47']:
        keys=sorted(k for k in days if k[0]==farm);assert len(keys)==180
        sums=[];counts=[]
        for i in range(0,len(keys),5):
            sub=[r for key in keys[i:i+5] for r in days[key]]
            sums.append(math.fsum((r['cc']-r['yy'])**2-(r['bb']-r['yy'])**2 for r in sub));counts.append(len(sub))
        ix=rng.integers(len(sums),size=(20000,len(sums)));ss+=np.asarray(sums)[ix].sum(1);nn+=np.asarray(counts)[ix].sum(1)
    sample=ss/nn;actual=dict(seed=seed,p_worse=float(np.mean(sample>=0)),ci=np.quantile(sample,[report['alpha'],1-report['alpha']]).tolist())
    saved=report['bootstrap'][str(seed)];assert actual['p_worse']==saved['p_worse'];compare(actual['ci'],saved['ci'],'bootstrap_ci')
    bootstrap.append(actual)

profile=[]
for v in ['DIAG10','A','B','EXT10','EXT12']:
    cc=[r for r in cells if r['validator']==v]
    profile.append(dict(validator=v,cells=len(cc),zero_weights=sum(r['weight']==0 for r in cc),one_weights=sum(r['weight']==1 for r in cc),min_weight=min(r['weight'] for r in cc),max_weight=max(r['weight'] for r in cc)))
first=json.loads((H/'first_fold_verification_v2.json').read_text(encoding='utf-8'));assert first['status']=='PASS'
assert all(r['maxdiff']==0 for r in first['replay']) and first['batch_maxdiff']==0
record=dict(status='PASS',scope='Completed fixed public CSV/inner NPZ arithmetic replay only; no model fit, oracle weight selection or raw target/EL1/test access',
            outer_rows=len(seen),cells=len(cells),source_sha256=source,max_gaps=dict(gaps),cell_checks=cells,weight_profile=profile,
            scores=scorecheck,segments=segmentcheck,bootstrap=bootstrap,alpha=report['alpha'],csv_sha256=csvhash,inner_npz_sha256=npzhash,
            parent_saved_checks_not_repeated_in_full=report['checks'],first_fold_replay_report_read_not_refit=True)
path=OUT/'completed_nested_log_check_v1.json';assert not path.exists()
path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k in ['status','outer_rows','cells','max_gaps','weight_profile','bootstrap']},ensure_ascii=False,indent=2))
print(json.dumps([r for r in cells if r['validator']=='B' and r['fold']==2],ensure_ascii=False,indent=2))
