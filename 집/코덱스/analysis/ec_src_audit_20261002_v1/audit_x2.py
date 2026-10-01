"""X2 저장 산출물 독립 검산 및 사전등록 누락 진단 보완. 모델 재학습 없음."""
from pathlib import Path
import os, sys
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'집/클로드/research'))
import env
import csv, json, math, hashlib, importlib.util
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
X2 = ROOT/'집/코덱스/analysis/ec_src_X2_20261002_v1'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def rmse(errors):
    return math.sqrt(math.fsum(e*e for e in errors)/len(errors))

def mean(values):
    return math.fsum(values)/len(values)

def dump(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    lockpath = Path(env.CODEX)/'ec_final_lock/locked_days.json'
    lock = {(r['farm'], int(r['day'])) for r in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
    raw = {}; skipped = 0
    for r in read(Path(env.DATA)/'train_y.csv'):
        f, d, h = r['row_id'].split('_')
        if f not in ['F13', 'F47']:
            continue
        if (f, int(d)) in lock:
            skipped += 1
            continue
        raw[r['row_id']] = float(r['sub_ec'])
    shape = read(X2/'shape_oof.csv')
    level = read(X2/'level_oof.csv')
    stored = json.loads((X2/'result.json').read_text(encoding='utf-8'))
    assert len(raw) == len(shape) == 8640 and skipped == 960
    assert {r['row_id'] for r in shape} == set(raw)
    days = defaultdict(list)
    for r in shape:
        key = (r['farm'], int(r['day']))
        assert key not in lock
        assert abs(float(r['sub_ec'])-raw[r['row_id']]) < 1e-12
        days[key].append(r)
    assert len(days) == len(level) == 360
    for key, rs in days.items():
        assert sorted(int(r['hour']) for r in rs) == list(range(24))
        lev = mean([raw[r['row_id']] for r in rs])
        assert max(abs(float(r['shape'])-(raw[r['row_id']]-lev)) for r in rs) < 1e-12
    levelmap = {(r['farm'], int(r['day'])):r for r in level}
    for key, r in levelmap.items():
        assert key in days
        lev = mean([raw[z['row_id']] for z in days[key]])
        assert abs(float(r['level'])-lev) < 1e-12
        assert len({z['fold'] for z in days[key]}) == 1
        assert r['fold'] == days[key][0]['fold'] and r['block'] == days[key][0]['block']
    errors = [float(r['v2'])-raw[r['row_id']] for r in shape]
    e_mean = {k:mean([float(r['v2'])-raw[r['row_id']] for r in rs]) for k, rs in days.items()}
    shape_errors = [(float(r['v2'])-raw[r['row_id']])-e_mean[(r['farm'], int(r['day']))] for r in shape]
    baseline = {'overall':rmse(errors), 'level':rmse(list(e_mean.values())), 'shape':rmse(shape_errors)}
    assert all(abs(baseline[k]-stored['baseline'][k]) < 1e-12 for k in baseline)
    assert abs(baseline['overall']**2-baseline['level']**2-baseline['shape']**2) < 1e-12
    high = {k for k, r in levelmap.items() if float(r['level']) >= 1.2}
    smodels = ['I', 'A', 'C', 'T0', 'T1', 'T3']
    lmodels = ['L-T', 'L-I', 'L-A', 'L-C']
    report = {'baseline':baseline, 'rows':len(shape), 'days':len(days), 'high_ec_threshold':1.2,
              'high_ec_days':len(high), 'locked_rows_excluded_before_float':skipped,
              'identity_max_error':abs(baseline['overall']**2-baseline['level']**2-baseline['shape']**2),
              'models':{}, 'raw_label_alignment':'PASS', 'final_lock_scored':False, 'test_X_read':False,
              'note':'projection은 사후 점수조건 대조이며 사전등록 원판정 불변; 인과 예측변환으로 쓰지 않음'}
    projected = {}
    for model in smodels:
        mday = {k:mean([float(r[model]) for r in rs]) for k, rs in days.items()}
        er = [float(r[model])-float(r['shape']) for r in shape]
        ep = [float(r[model])-mday[(r['farm'],int(r['day']))]-float(r['shape']) for r in shape]
        projected[model] = ep
        assert abs(rmse(er)-stored['models'][model]['rmse']) < 1e-12
        assert abs(rmse(er)**2-rmse(ep)**2-mean([z*z for z in mday.values()])) < 1e-12
        rec = {'raw_shape_rmse':rmse(er), 'projected_shape_rmse':rmse(ep),
               'prediction_daily_mean_rmse':rmse(list(mday.values())), 'strata':{}}
        for label, mask in [
            ('00_06',[int(r['hour']) <= 6 for r in shape]),
            ('07_23',[int(r['hour']) > 6 for r in shape]),
            ('high_ec',[ (r['farm'],int(r['day'])) in high for r in shape]),
            ('nonhigh_ec',[ (r['farm'],int(r['day'])) not in high for r in shape])]:
            rr = [a for a,b in zip(er,mask) if b]; pp = [a for a,b in zip(ep,mask) if b]
            vv = [a for a,b in zip(shape_errors,mask) if b]
            rec['strata'][label] = {'rows':len(rr), 'raw_shape_rmse':rmse(rr),
                                   'projected_shape_rmse':rmse(pp),'v2_projected_shape_rmse':rmse(vv)}
        report['models'][model] = rec
    for model in lmodels:
        er = [float(r[model])-float(r['level']) for r in level]
        assert abs(rmse(er)-stored['models'][model]['rmse']) < 1e-12
        report['models'][model] = {'level_rmse':rmse(er),'strata':{}}
        for label, mask in [('high_ec',[(r['farm'],int(r['day'])) in high for r in level]),
                            ('nonhigh_ec',[(r['farm'],int(r['day'])) not in high for r in level])]:
            rr=[a for a,b in zip(er,mask) if b]
            report['models'][model]['strata'][label]={'days':len(rr),'level_rmse':rmse(rr)}
    # Reproduce the fixed 20,000 block-bootstrap draws with independent fsum block aggregation.
    rng = np.random.default_rng(261002)
    draws = {}
    for farm in ['F13','F47']:
        blocks = sorted({int(r['block']) for r in level if r['farm'] == farm})
        draws[farm] = (blocks,rng.integers(0,len(blocks),size=(20000,len(blocks))))
    for model in smodels+lmodels:
        frame = shape if model in smodels else level
        target = 'shape' if model in smodels else 'level'
        metric_errors=[float(r[model])-float(r[target]) for r in frame]
        variants={'raw':metric_errors}
        if model in smodels:variants['projected']=projected[model]
        for metric, errs in variants.items():
            total_n=np.zeros(20000);total_ss=np.zeros(20000)
            for farm in ['F13','F47']:
                blocks, indices=draws[farm]
                groups={b:[e for r,e in zip(frame,errs) if r['farm']==farm and int(r['block'])==b] for b in blocks}
                block_n=np.array([len(groups[b]) for b in blocks],dtype=float)
                block_ss=np.array([math.fsum(e*e for e in groups[b]) for b in blocks])
                total_n+=block_n[indices].sum(axis=1);total_ss+=block_ss[indices].sum(axis=1)
            samples=np.sqrt(total_ss/total_n)
            ci=np.quantile(samples,[.0025,.9975]).tolist()
            threshold=.05 if model in smodels else .1
            report['models'][model][metric+'_bootstrap_ci995']=ci
            report['models'][model][metric+'_bootstrap_threshold_pass_fraction']=float(np.mean(samples<=threshold))
            if metric=='raw':
                assert max(abs(a-b) for a,b in zip(ci,stored['models'][model]['ci995'])) < 1e-12
    # Check implementation features against an independent sequential construction.
    spec=importlib.util.spec_from_file_location('x2_registered',X2/'run.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    records=[]
    for r in read(Path(env.DATA)/'train_X.csv'):
        f,d,h=r['row_id'].split('_')
        if f not in ['F13','F47']:continue
        rec={'row_id':r['row_id'],'farm':f,'day':int(d),'hour':int(h)}
        for c in mod.RAW:
            rec[c]=float(r[c]) if r[c].strip().lower() not in ['','nan','na','none','null'] else math.nan
        records.append(rec)
    allx=pd.DataFrame(records).sort_values(['farm','day','hour']).reset_index(drop=True)
    actual=mod.features(allx)
    maxdiff=0.0
    for _,group in allx.groupby(['farm','day'],sort=False):
        sequence=group.to_dict('records'); cum={c:[] for c in ['in_temp']+mod.ACT}
        for j,(idx,row) in enumerate(zip(group.index,sequence)):
            hh=row['hour']; cc={'day':row['day'],'hour':hh}
            for k in [1,2]:cc['sin'+str(k)]=math.sin(hh*math.pi*k/12);cc['cos'+str(k)]=math.cos(hh*math.pi*k/12)
            ii=dict(cc);ii['temp']=row['in_temp'];ii['temp2']=row['in_temp']**2
            for c in cum:
                if math.isfinite(row[c]):cum[c].append(row[c])
            ii['h0']=sequence[0]['in_temp'];ii['cumtemp']=mean(cum['in_temp']) if cum['in_temp'] else math.nan
            for lag in [1,3]:
                z=sequence[j-lag]['in_temp'] if j>=lag else math.nan
                ii['lag'+str(lag)]=z if math.isfinite(z) else ii['h0']
            ii['d0']=ii['temp']-ii['h0'];aa=dict(cc)
            for c in mod.ACT:
                aa[c]=row[c];aa[c+'_h0']=sequence[0][c];aa[c+'_cum']=mean(cum[c]) if cum[c] else math.nan
            for name, vals in [('C',cc),('I',ii),('A',aa)]:
                for col,value in vals.items():
                    v=float(actual[name].at[idx,col])
                    if math.isnan(value):assert math.isnan(v)
                    else:
                        maxdiff=max(maxdiff,abs(v-value));assert math.isclose(v,value,rel_tol=1e-12,abs_tol=1e-10)
    causal_checks=[]
    for farm in ['F13','F47']:
        day=int(allx.loc[allx.farm.eq(farm),'day'].median())
        for hour in [0,6,12,23]:
            prefix=allx.farm.eq(farm)&((allx.day*24+allx.hour)<=day*24+hour)
            changed=allx.copy();changed.loc[~prefix,mod.RAW]=changed.loc[~prefix,mod.RAW]*11+1234
            alternate=mod.features(changed)
            for name in ['I','A','C']:
                pd.testing.assert_frame_equal(actual[name].loc[prefix],alternate[name].loc[prefix])
            causal_checks.append({'farm':farm,'cutoff_day':day,'cutoff_hour':hour,'prefix_rows':int(prefix.sum()),'status':'PASS'})
    report['feature_manual_max_abs_difference']=maxdiff
    report['causality_checks']=causal_checks
    report['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), X2/'PROTOCOL.md',X2/'run.py',X2/'shape_oof.csv',X2/'level_oof.csv',X2/'result.json',Path(env.DATA)/'train_y.csv',Path(env.DATA)/'train_X.csv',lockpath]}
    dump('x2_audit.json',report)
    print(json.dumps({'baseline':baseline,'rows':len(shape),'days':len(days),'high_ec_days':len(high),
                      'all_saved_metrics_and_bootstrap_ci_agree':True,'manual_feature_max_abs_difference':maxdiff,
                      'causality_checks_pass':len(causal_checks),
                      'models':{k:{q:v for q,v in r.items() if q in ['raw_shape_rmse','projected_shape_rmse','level_rmse','raw_bootstrap_threshold_pass_fraction','projected_bootstrap_threshold_pass_fraction']} for k,r in report['models'].items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
