"""EL1 저장 결과의 숫자 및 검증 배치 독립 감사. fit 없음."""
from pathlib import Path
import sys,csv,json,math,importlib.util,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
source=ROOT/'집/클로드/research/local/ec2_EL1_oof.csv'
spec=importlib.util.spec_from_file_location('el1_readonly_common',HERE.parent/'ec_model_common_20261002_v1/common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rms(rows,column):return math.sqrt(math.fsum((r[column]-r['y'])**2 for r in rows)/len(rows))

def main():
    raw,lab,_,locks,core=c.prepare();labels=dict(zip(lab.row_id,lab.sub_ec))
    folds=[]
    for farm in ['F13','F47']:
        days=sorted(lab.loc[lab.farm.eq(farm)&lab.day.ge(179),'day'].unique().tolist())
        for j in range(0,len(days),5):folds.append(('EL1',len(folds),{(farm,int(d)) for d in days[j:j+5]}))
    with source.open(encoding='utf-8-sig',newline='') as stream:rawrows=list(csv.DictReader(stream))
    rows=[]
    for r in rawrows:
        f,day,hour=r['row_id'].split('_');day=int(day);hour=int(hour)
        assert (f,day) not in locks and r['row_id'] in labels and day>=179
        y=float(r['sub_ec']);assert y==labels[r['row_id']]
        row={'id':r['row_id'],'farm':f,'day':day,'hour':hour,'fold':int(r['fold']),'gap':int(r['gap']),'y':y}
        for s in [7,101,2024]:
            row[f'day_{s}']=float(r[f'r3_{s}']);row[f'season_{s}']=float(r[f'r3s_{s}'])
        rows.append(row)
    assert len(rows)==1104 and len({r['id'] for r in rows})==1104 and len({(r['farm'],r['day']) for r in rows})==46
    assert len(folds)==10 and {r['fold'] for r in rows}==set(range(10))
    geometry=[]
    for fold in folds:
        tr,va=c.split_fold(raw,lab,fold,locks)
        g=[r for r in rows if r['fold']==fold[1]]
        assert {r['id'] for r in g}==set(va.row_id) and len(g)==len(va)
        for r in g:
            td=tr.loc[tr.farm.eq(r['farm'])&tr.day.ge(179),'day'].unique()
            gap=min(abs(int(x)-r['day']) for x in td)
            assert gap==r['gap'] and gap>=2
        geometry.append({'fold':fold[1],'farm':g[0]['farm'],'n_rows':len(g),'days':len(fold[2]),'gap_min':min(r['gap'] for r in g),'gap_max':max(r['gap'] for r in g)})
    scores=[];groups=[('전체',rows)]+[(f,[r for r in rows if r['farm']==f]) for f in ['F13','F47']]
    groups += [('이웃6일이내',[r for r in rows if r['gap']<=6]),('이웃6일초과',[r for r in rows if r['gap']>6])]
    maxgap=0.
    for label,g in groups:
        for seed in [7,101,2024]:
            a=rms(g,f'day_{seed}');b=rms(g,f'season_{seed}')
            for key,val in [(f'day_{seed}',a),(f'season_{seed}',b)]:
                other=float(np.sqrt(np.mean([(r[key]-r['y'])**2 for r in g])))
                gap=abs(val-other);maxgap=max(maxgap,gap);assert gap<1e-12
            scores.append({'group':label,'seed':seed,'rows':len(g),'days':len({(r['farm'],r['day']) for r in g}),
                           'day_r3_rmse':a,'season_r3_rmse':b,'change_pct':100*(b/a-1),'improved':b<a})
    report={'status':'PASS','source_oof_sha256':sha(source),'source_code_sha256':sha(ROOT/'집/클로드/research/ec2_EL1_evallike_late_blocks_v1.py'),
            'rows':1104,'days':46,'folds':geometry,'scores':scores,'max_independent_rmse_gap':maxgap,
            'final_lock_scored':False,'new_model_trained':False,'adoption_rule_changed':False,'cal_values_used':False,
            'confidence':'산술·행집합·거리 재현 높음, 숨은 평가 일반화 해석 중간',
            'limitations':['후반46일의 공개 진단','R3 단독이며 통합 v2 아님','같은 검증 데이터 반복 사용','평가 개선 및 후보 채택 확정 근거 아님']}
    with (HERE/'audit_el1_v1.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    lines=['# EL1 저장 결과 독립 감사 · 사후 진단','',
           '잠금 제외 후반46일1104행의 10개 검증 묶음, 검증/잠금 ±1 purge와 최단 학습정답일 거리를 독립 재구성해 저장 결과와 일치했다. CSV 및 math.fsum/NumPy RMSE도 일치했다. 새로운 학습이나 채택 기준 변경은 없다.','',
           '| 그룹 | 시드 | 행 | day R3 | 계절 R3 | 변화% |','|---|---|---|---|---|---|']
    lines += [f"| {r['group']} | {r['seed']} | {r['rows']} | {r['day_r3_rmse']:.9f} | {r['season_r3_rmse']:.9f} | {r['change_pct']:+.3f} |" for r in scores]
    lines += ['', '신뢰도는 산술 및 검증 배치 재현에 대해 높음, 숨은 평가 일반화에 대해 중간이다. 이 검토는 기존 R3 진단이며 요청서의 전체 v2·모든 검증기·잠금 단회 기준을 대체하지 않는다. cal/deep_cal_9 값은 읽어 사용하지 않았다.']
    with (HERE/'검토보고서_v1.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':'PASS','pooled':[r for r in scores if r['group']=='전체'],'max_rmse_gap':maxgap},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
