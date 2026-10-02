"""독립 검산된 v2 OOF가 준비되면 후반 구간을 명시적으로 보고. 새 판정 없음."""
from pathlib import Path
import csv,json,math,time,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LOCAL=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
ARMS=['v2','season_v2','day_r3','season_r3']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def calculate(rows,arm):
    errors=[r[arm]-r['y'] for r in rows];mse=math.fsum(x*x for x in errors)/len(rows)
    days={}
    for r,error in zip(rows,errors):days.setdefault((r['farm'],r['day']),[]).append(error)
    level=shape=0.
    for group in days.values():
        assert len(group)==24
        bias=math.fsum(group)/24
        level+=24*bias*bias;shape+=math.fsum((r-bias)**2 for r in group)
    assert abs(mse-(level+shape)/len(rows))<1e-12
    return {'rmse':math.sqrt(mse),'mean_bias':math.fsum(errors)/len(rows),
            'level_rmse':math.sqrt(level/len(rows)),'shape_rmse':math.sqrt(shape/len(rows)),
            'level_squared_error_share':level/(level+shape),'n':len(rows),'days':len(days)}

def main():
    marker=HERE/'v2_integration_independent_verification.json';deadline=time.monotonic()+48*3600
    while not marker.exists():
        assert time.monotonic()<deadline,'v2 독립 검산 결과 대기 48시간 초과. 큐 상태 점검 필요.'
        time.sleep(10)
    evidence=json.loads(marker.read_text(encoding='utf-8'));path=LOCAL/'v2_integration_oof.csv'
    assert evidence['status']=='PASS' and evidence['csv_sha256']==sha(path)
    grouped={}
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for r in csv.DictReader(stream):
            if r['validator']!='DIAG10':continue
            key=r['row_id'];f,day,hour=key.split('_')
            record=grouped.setdefault(key,{'farm':f,'day':int(day),'hour':int(hour),'ys':[],
                                           'seeds':[],**{a:[] for a in ARMS}})
            record['ys'].append(float(r['sub_ec']));record['seeds'].append(int(r['seed']))
            for arm in ARMS:record[arm].append(float(r[arm]))
    assert len(grouped)==8640
    rows=[]
    for key,r in sorted(grouped.items()):
        assert sorted(r['seeds'])==[7,101,2024] and len(set(r['ys']))==1
        rows.append({'id':key,'farm':r['farm'],'day':r['day'],'hour':r['hour'],'y':r['ys'][0],
                     **{a:math.fsum(r[a])/3 for a in ARMS}})
    groups=[('전체',rows),('전반(day<179)',[r for r in rows if r['day']<179]),
            ('후반(day>=179)',[r for r in rows if r['day']>=179])]
    groups += [(f+' 후반',[r for r in rows if r['farm']==f and r['day']>=179]) for f in ['F13','F47']]
    assert len(groups[1][1])==314*24 and len(groups[2][1])==46*24
    scores=[]
    for label,g in groups:
        for arm in ARMS:scores.append({'group':label,'arm':arm,**calculate(g,arm)})
    summary=json.loads((HERE/'v2_integration_result.json').read_text(encoding='utf-8'))
    for arm in ['v2','season_v2']:
        old=next(r['rmse'] for r in summary['ensemble'] if r['validator']=='DIAG10' and r['arm']==arm)
        new=next(r['rmse'] for r in scores if r['group']=='전체' and r['arm']==arm)
        assert abs(old-new)<1e-12
    result={'status':'PASS','aggregation':'각 행 3시드 예측 평균 후 RMSE; 진단 전용',
            'source_sha256':sha(path),'scores':scores,'adoption_rule_changed':False,
            'final_lock_labels_read':False,'limitation':'후반46일은 공개 검증 표본이며 숨은 평가 개선을 보증하지 않는다.'}
    with (HERE/'후반_구간검산_v1.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    text=['# DC4 v2 후반 구간 독립 검산','',
          '독립 검산을 통과한 DIAG10 OOF에서 각 행의 세 시드 예측을 먼저 평균했다. 전체360일8640행, 전반314일7536행, 후반46일1104행이다. 이 표는 후반 결과 서술이며 모든 시드×검증기 채택 기준과 단회 잠금 확인을 대체하지 않는다.','',
          '| 구간 | 모델 | 일수 | RMSE | 수준 RMSE | 모양 RMSE | 평균 잔차 |','|---|---|---|---|---|---|---|']
    text += [f"| {r['group']} | {r['arm']} | {r['days']} | {r['rmse']:.9f} | {r['level_rmse']:.9f} | {r['shape_rmse']:.9f} | {r['mean_bias']:+.9f} |" for r in scores]
    text += ['', '하루 평균 잔차의 제곱과 하루 안에서 중심화한 잔차의 제곱 합이 전체 MSE와 일치함을 확인했다. 신뢰도는 산술 재현에 대해 높음, 숨은 평가 일반화에 대해 중간이다. 공개 검증 반복 사용과 실제 평가 배치 차이 가능성은 남는다. 원시 잠금 라벨은 읽지 않았다.']
    with (HERE/'후반_구간검산_v1.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(text)+'\n')
    print(json.dumps({'status':'PASS','late':[r for r in scores if r['group']=='후반(day>=179)']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
