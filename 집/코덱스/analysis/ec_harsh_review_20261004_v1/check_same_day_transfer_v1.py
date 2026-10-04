"""Rebuild a descriptive same-day diagnostic from fixed public CSV only."""
from pathlib import Path
import csv,json,math,hashlib,statistics
from collections import defaultdict
from decimal import Decimal,localcontext
ROOT=Path(__file__).resolve().parents[4]
H=ROOT/'집/코덱스/analysis/ec_followup_progress_20261004_v1'
SRC=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
OUT=Path(__file__).resolve().parent
def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def avg(values):
    values=list(values);return math.fsum(values)/len(values)
def close(a,b):assert abs(a-b)<1e-12,(a,b)
FIELDS=['x','yday','required_shift','learned_correction']
report=json.loads((H/'same_day_transfer_v1.json').read_text(encoding='utf-8'))
raw=table(SRC/'full_shift_records_v1.csv');assert len(raw)==31236
assert sha(SRC/'full_shift_records_v1.csv')==report['input_sha256']
assert sha(H/'diagnose_same_day_transfer_v1.py')==report['code_sha256']
seen=set(); selected=[]; byday=defaultdict(list)
for r in raw:
    for c in ['fold','seed','hour','day']:r[c]=int(r[c])
    for c in FIELDS:r[c]=float(r[c]);assert math.isfinite(r[c])
    key=tuple(r[c] for c in ['validator','fold','seed','hour','domain','farm','day'])
    assert key not in seen;seen.add(key)
    close(r['required_shift'],r['yday']-r['x'])
    if r['validator']=='DIAG10' and r['hour'] in [0,23]:
        selected.append(r)
        byday[r['seed'],r['hour'],r['domain'],r['farm'],r['day']].append(r)
daily={};decimal_diff=0.
for key,rr in byday.items():
    assert max(r['yday'] for r in rr)-min(r['yday'] for r in rr)<1e-12
    seed,h,domain,farm,day=key
    daily[key]=dict(seed=seed,hour=h,domain=domain,farm=farm,day=day,occurrences=len(rr),**{c:avg(r[c] for r in rr) for c in FIELDS})
    if domain=='outer':assert len(rr)==1
    with localcontext() as context:
        context.prec=45
        for c in FIELDS:
            alternative=float(sum((Decimal(r[c]) for r in rr),Decimal(0))/Decimal(len(rr)))
            decimal_diff=max(decimal_diff,abs(alternative-daily[key][c]))
assert decimal_diff<1e-12

def stats(rr):
    if not rr:return {'n':0}
    return dict(n=len(rr),unique_days=len({(r['farm'],r['day']) for r in rr}),
                **{c:avg(r[c] for r in rr) for c in FIELDS},
                actual_high_n=sum(r['yday']>=1 for r in rr),
                actual_high_fraction=avg(r['yday']>=1 for r in rr))
def verify(actual,expected):
    assert set(actual)==set(expected),(actual,expected)
    for col,value in actual.items():
        if isinstance(value,float):close(value,float(expected[col]))
        else:assert value==expected[col]
repeats=[];decomposition=[];strata=[];pairs=[];overlaps=[];movements=[]
for seed in [7,101,2024]:
    for h in [0,23]:
        occurrence=[r for r in selected if r['seed']==seed and r['hour']==h]
        dd=[r for key,r in daily.items() if key[:2]==(seed,h)]
        for domain in ['inner','outer']:
            occ=[r for r in occurrence if r['domain']==domain and r['x']>=.9]
            day=[r for r in dd if r['domain']==domain and r['x']>=.9]
            actual=dict(seed=seed,hour=h,domain=domain,occurrence_selected=stats(occ),equal_day_mean_x_selected=stats(day))
            expected=next(r for r in report['occurrence_vs_equal_day'] if (r['seed'],r['hour'],r['domain'])==(seed,h,domain))
            verify(actual['occurrence_selected'],expected['occurrence_selected'])
            verify(actual['equal_day_mean_x_selected'],expected['equal_day_mean_x_selected'])
            repeats.append(actual)
            for farm in ['F13','F47']:
                for phase in ['early','late']:
                    rr=[r for r in day if r['farm']==farm and (r['day']>=179)==(phase=='late')]
                    strata.append(dict(seed=seed,hour=h,domain=domain,farm=farm,phase=phase,**stats(rr)))
            if domain=='inner':
                # Keep the .9 occurrence selection fixed, then vary only repeat weights.
                fixed=defaultdict(list)
                for r in occ:fixed[r['farm'],r['day']].append(r)
                fixed_days=[dict(farm=f,day=d,**{c:avg(r[c] for r in rr) for c in FIELDS}) for (f,d),rr in fixed.items()]
                all_occ_same_days=[r for r in dd if r['domain']=='inner' and (r['farm'],r['day']) in fixed]
                decomposition.append(dict(seed=seed,hour=h,
                    occurrence_selected=stats(occ),
                    same_selected_occurrences_equal_day=stats(fixed_days),
                    same_selected_days_all_occurrences_equal_day=stats(all_occ_same_days),
                    averaged_x_reselection_equal_day=stats(day)))
        inner={(r['farm'],r['day']):r for r in dd if r['domain']=='inner'}
        outer={(r['farm'],r['day']):r for r in dd if r['domain']=='outer'}
        assert len(inner)==217 and len(outer)==360 and set(inner)<=set(outer)
        for (farm,d),ri in inner.items():
            ro=outer[farm,d];close(ri['yday'],ro['yday'])
            ih,oh=ri['x']>=.9,ro['x']>=.9
            membership='both_high' if ih and oh else ('inner_only_high' if ih else ('outer_only_high' if oh else 'neither_high'))
            item=dict(seed=seed,hour=h,farm=farm,day=d,membership=membership,
                      x_delta=ro['x']-ri['x'],needed_shift_delta=ro['required_shift']-ri['required_shift'])
            for suffix,r in [('inner',ri),('outer',ro)]:
                for c in FIELDS+['occurrences']:item[c+'_'+suffix]=r[c]
            close(item['x_delta'],-item['needed_shift_delta']);pairs.append(item)
        ih={k for k,r in inner.items() if r['x']>=.9};oh={k for k,r in outer.items() if r['x']>=.9}
        overlap=dict(seed=seed,hour=h,inner_coverage=len(inner),outer_coverage=len(outer),inner_high=len(ih),outer_high=len(oh),
                     common=len(ih&oh),inner_only=len(ih-oh),outer_only=len(oh-ih))
        verify(overlap,next(r for r in report['overlap'] if (r['seed'],r['hour'])==(seed,h)))
        overlap.update(outer_only_with_inner_low=len((oh-ih)&set(inner)),outer_high_without_inner_coverage=len(oh-set(inner)),
                       paired_true_high=sum(r['yday']>=1 for r in inner.values()),
                       outer_true_high=sum(r['yday']>=1 for r in outer.values()))
        assert overlap['outer_only']==overlap['outer_only_with_inner_low']+overlap['outer_high_without_inner_coverage']
        overlaps.append(overlap)
        pp=[r for r in pairs if (r['seed'],r['hour'])==(seed,h)]
        for group in ['all','actual_high','actual_ordinary','both_high','inner_only_high','outer_only_high','neither_high']:
            gg=[r for r in pp if group=='all' or (group=='actual_high' and r['yday_inner']>=1) or (group=='actual_ordinary' and r['yday_inner']<1) or r['membership']==group]
            result=dict(seed=seed,hour=h,group=group,n=len(gg))
            if gg:
                for c in ['x_inner','x_outer','x_delta','required_shift_inner','required_shift_outer','needed_shift_delta','learned_correction_inner','learned_correction_outer']:
                    result[c]=avg(r[c] for r in gg)
                result['x_delta_median']=statistics.median(r['x_delta'] for r in gg)
                result['outer_prediction_higher_n']=sum(r['x_delta']>0 for r in gg)
            verify(result,next(r for r in report['movement'] if (r['seed'],r['hour'],r['group'])==(seed,h,group)))
            movements.append(result)

saved_pairs=table(SRC/'same_day_transfer_pairs_v1.csv');assert len(pairs)==len(saved_pairs)==1302
saved_seen=set();pair_maxdiff=0.
pair_map={(r['seed'],r['hour'],r['farm'],r['day']):r for r in pairs}
for r in saved_pairs:
    key=int(r['seed']),int(r['hour']),r['farm'],int(r['day'])
    assert key in pair_map and key not in saved_seen;saved_seen.add(key);a=pair_map[key]
    assert r['membership']==a['membership']
    for c,value in a.items():
        if c in ['seed','hour','farm','day','membership']:continue
        if c.startswith('occurrences_'):assert int(r[c])==value
        else:pair_maxdiff=max(pair_maxdiff,abs(float(r[c])-value))
assert pair_maxdiff<1e-12

for filename,expected,keys in [('same_day_transfer_strata_v1.csv',strata,['seed','hour','domain','farm','phase']),('same_day_transfer_movement_v1.csv',movements,['seed','hour','group'])]:
    rows=table(H/filename);assert len(rows)==len(expected)
    matched=set()
    for r in rows:
        for c in ['seed','hour']:r[c]=int(r[c])
        key=tuple(r[c] for c in keys);assert key not in matched;matched.add(key)
        a=next(z for z in expected if tuple(z[c] for c in keys)==key)
        for c,value in a.items():
            if c in keys:assert r[c]==value
            elif isinstance(value,float):close(float(r[c]),value)
            else:assert int(r[c])==value

record=dict(status='PASS',scope='Posthoc descriptive public same-day statistics only; no prediction scoring, fit, new threshold or causal inference',
            input_rows=len(raw),selected_occurrence_rows=len(selected),daily_rows=len(daily),paired_rows=len(pairs),
            unique_inner_days=217,unique_outer_days=360,paired_true_high_days=20,total_true_high_days=31,
            decimal_daily_maxdiff=decimal_diff,stored_pair_maxdiff=pair_maxdiff,
            occurrence_vs_equal_day=repeats,weight_membership_decomposition=decomposition,
            overlap=overlaps,movement=movements,strata=strata,
            input_sha256=sha(SRC/'full_shift_records_v1.csv'),paired_csv_sha256=sha(SRC/'same_day_transfer_pairs_v1.csv'))
path=OUT/'same_day_transfer_check_v1.json';assert not path.exists()
path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ['occurrence_vs_equal_day','movement','strata','weight_membership_decomposition']},ensure_ascii=False,indent=2))
print(json.dumps([dict(seed=r['seed'],hour=r['hour'],**{c:dict(n=r[c]['n'],required_shift=r[c]['required_shift']) for c in ['occurrence_selected','same_selected_occurrences_equal_day','same_selected_days_all_occurrences_equal_day','averaged_x_reselection_equal_day']}) for r in decomposition],ensure_ascii=False,indent=2))
