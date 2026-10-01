"""X2 로그 생성식: csv/math.fsum/math.exp로 계수·예측·점수 독립 재구성."""
from pathlib import Path
import os, sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,math,hashlib
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'집/코덱스/analysis/ec_src_X2_loglaw_20261002_v2'

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def mean(v):
    v=[z for z in v if math.isfinite(z)]
    return math.fsum(v)/len(v) if v else math.nan
def rmse(v):return math.sqrt(math.fsum(z*z for z in v)/len(v))

def main():
    lockpath=Path(env.CODEX)/'ec_final_lock/locked_days.json'
    locks={(r['farm'],int(r['day'])) for r in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
    raw={};skip=0
    for r in read(Path(env.DATA)/'train_y.csv'):
        f,d,h=r['row_id'].split('_')
        if f not in ['F13','F47']:continue
        if (f,int(d)) in locks:skip+=1;continue
        raw[r['row_id']]={'ec':float(r['sub_ec']),'temp':float(r['sub_temp'])}
    for r in read(Path(env.DATA)/'train_X.csv'):
        if r['row_id'] in raw:raw[r['row_id']]['in_temp']=float(r['in_temp']) if r['in_temp'].strip() else math.nan
    rows=read(SOURCE/'predictions.csv');result=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    coeff={(r['model'],int(r['fold'])):float(r['beta']) for r in read(SOURCE/'coefficients.csv')}
    assert len(rows)==len(raw)==8640 and skip==960 and len(coeff)==60
    days=defaultdict(list)
    for i,r in enumerate(rows):
        assert r['row_id'] in raw
        for col,rawcol in [('sub_ec','ec'),('sub_temp','temp'),('in_temp','in_temp')]:
            vv=float(r[col]) if r[col].strip() else math.nan;zz=raw[r['row_id']][rawcol]
            assert (math.isnan(vv) and math.isnan(zz)) or abs(vv-zz)<1e-12
        days[(r['farm'],int(r['day']))].append(i)
    assert len(days)==360 and not (set(days)&locks)
    centerlog={};levels={}
    max_daymean=0.;max_targetdiff=0.
    for key,ids in days.items():
        ids.sort(key=lambda i:int(rows[i]['hour']))
        assert [int(rows[i]['hour']) for i in ids]==list(range(24))
        vv=[raw[rows[i]['row_id']]['ec'] for i in ids];levels[key]=mean(vv)
        ll=[math.log(v) for v in vv];mm=mean(ll)
        for i,v in zip(ids,ll):
            centerlog[i]=v-mm
            max_targetdiff=max(max_targetdiff,abs(centerlog[i]-float(rows[i]['centerlog'])))
    assert max_targetdiff<1e-12
    rng=np.random.default_rng(2610022);draws={}
    for farm in ['F13','F47']:
        blocks=sorted({int(r['block']) for r in rows if r['farm']==farm})
        draws[farm]=(blocks,rng.integers(0,len(blocks),size=(20000,len(blocks))))
    answer={'rows':8640,'days':360,'locked_rows_excluded_before_float':skip,'raw_label_alignment':'PASS',
            'oracle_only':True,'final_lock_scored':False,'test_X_read':False,'models':{},'max_logtarget_difference':max_targetdiff}
    max_betadiff=0.;max_preddiff=0.
    for col in ['sub_temp','in_temp']:
        for lag in [0,1,3]:
            name=f'{col}_lag{lag}';center={}
            for key,ids in days.items():
                vals=[raw[rows[i]['row_id']]['temp' if col=='sub_temp' else 'in_temp'] for i in ids]
                delayed=[vals[max(0,j-lag)] for j in range(24)]
                if lag:delayed=[v if math.isfinite(v) else vals[0] for v in delayed]
                mm=mean(delayed)
                center.update({i:t-mm for i,t in zip(ids,delayed)})
            rebuilt={};beta_table=[]
            for fold in range(10):
                vadays={k for k,ids in days.items() if int(rows[ids[0]]['fold'])==fold}
                forbidden={(f,d+j) for f,d in locks|vadays for j in [-1,0,1]}
                tr=[i for k,ids in days.items() if k not in forbidden for i in ids if math.isfinite(center[i])]
                beta=math.fsum(center[i]*centerlog[i] for i in tr)/math.fsum(center[i]**2 for i in tr)
                max_betadiff=max(max_betadiff,abs(beta-coeff[(name,fold)]))
                beta_table.append({'fold':fold,'training_rows':len(tr),'beta':beta})
                for key in vadays:
                    ids=days[key];z=[math.exp(beta*center[i]) if math.isfinite(center[i]) else 1.0 for i in ids];zm=mean(z)
                    for i,v in zip(ids,z):rebuilt[i]=levels[key]*v/zm
            assert len(rebuilt)==8640
            for i,r in enumerate(rows):max_preddiff=max(max_preddiff,abs(rebuilt[i]-float(r[name])))
            for key,ids in days.items():max_daymean=max(max_daymean,abs(mean([rebuilt[i] for i in ids])-levels[key]))
            err=[rebuilt[i]-raw[r['row_id']]['ec'] for i,r in enumerate(rows)]
            score=rmse(err);assert abs(score-result['models'][name]['rmse'])<1e-12
            n=np.zeros(20000);ss=np.zeros(20000)
            for farm in ['F13','F47']:
                blocks,indices=draws[farm]
                groups={b:[e for r,e in zip(rows,err) if r['farm']==farm and int(r['block'])==b] for b in blocks}
                bn=np.array([len(groups[b]) for b in blocks]);bs=np.array([math.fsum(e*e for e in groups[b]) for b in blocks])
                n+=bn[indices].sum(axis=1);ss+=bs[indices].sum(axis=1)
            samples=np.sqrt(ss/n);ci=np.quantile(samples,[.05/12,1-.05/12]).tolist();prob=float(np.mean(samples<=.05))
            assert max(abs(a-b) for a,b in zip(ci,result['models'][name]['ci_bonferroni6']))<1e-12
            assert prob==result['models'][name]['bootstrap_probability_rmse_le_005']
            answer['models'][name]={'rmse':score,'ci_bonferroni6':ci,'bootstrap_probability_rmse_le_005':prob,'coefficients':beta_table}
    assert max_preddiff<1e-12 and max_betadiff<1e-12 and max_daymean<1e-12
    answer.update(max_beta_difference=max_betadiff,max_prediction_difference=max_preddiff,max_daymean_error=max_daymean)
    answer['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),SOURCE/'run.py',SOURCE/'PROTOCOL.md',SOURCE/'predictions.csv',SOURCE/'coefficients.csv',SOURCE/'result.json']}
    (HERE/'x2_loglaw_audit.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in answer.items() if k!='hashes'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

