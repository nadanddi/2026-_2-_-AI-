"""Read-only source audit and immutable experiment registration; no fitting."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env

RAW = ['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2',
       'act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
SEEDS = [47,1414,6464]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def write(name, obj):
    p = HERE / name
    assert not p.exists(), p
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

cat = ROOT / '공용/데이터_단서_카탈로그.md'
text = cat.read_text(encoding='utf-8-sig')
records = []
for line in text.splitlines():
    if re.match(r'^\|\s*6\.\d+\s*\|',line):
        fields = line.split('|')
        records.append({'number':fields[1].strip(),'title':fields[2].strip(),'full_record':line})
write('catalog_snapshot_v1.json',{'sha256':sha(cat),'records':records,
    'latest_number':max(int(r['number'].split('.')[1]) for r in records)})
with (HERE/'catalog_reading_index_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['number','title']);w.writeheader()
    w.writerows({k:r[k] for k in ['number','title']} for r in records)

paths={n:Path(env.DATA)/n for n in ['train_X.csv','train_y.csv','test_X.csv']}
x,y,t=(read(paths[n]) for n in paths)
xi={r['row_id']:r for r in x};yi={r['row_id']:r for r in y}
assert len(xi)==len(x) and len(yi)==len(y) and not(set(yi)-set(xi))
counts={c:sum(bool(r[c].strip()) for r in y) for c in ['sub_ec','sub_temp']}
for col,count in counts.items():
    joined=sum(bool(yi[r['row_id']][col].strip()) for r in x if r['row_id'] in yi)
    grouped={}
    for r in y:
        farm=r['row_id'].split('_')[0]
        grouped[farm]=grouped.get(farm,0)+bool(r[col].strip())
    assert count==joined==sum(grouped.values())
masked=[c for c in t[0] if c!='row_id' and all(not r[c].strip() for r in t)]
assert set(RAW)==set(t[0])-set(masked)-{'row_id'}
write('source_contract_v1.json',{
    'SHA256':{n:sha(p) for n,p in paths.items()},'train_X_rows':len(x),'train_y_rows':len(y),
    'EC_label_rows':counts['sub_ec'],'temperature_label_rows':counts['sub_temp'],
    'temperature_labels_without_EC':sum(bool(r['sub_temp'].strip()) and not r['sub_ec'].strip() for r in y),
    'X_without_y':len(set(xi)-set(yi)),'farms':len(set(r['row_id'].split('_')[0] for r in x)),
    'primary_raw_inputs':RAW,'MASK_excluded':masked,
    'allowed_sources':['row_id 온실/상대기록일/시각','같은 온실 현재/이전 입력 prefix',
        '공개 학습 EC 및 온도 정답 참조','전체 온실 공개 입력/온도 보조학습 후보'],
    'limitations':['복원/변환 관측 표시 없음','일차 연속이 물리적 연속을 보장하지 않음',
        '평가 실제 온도 없음','양 정답 검증 보류','현재 데이터의 예측 효과는 별도 검증 필요'],
    'verification':'정답 수: 직접 y / X ID 결합 / 온실 합계 일치',
})

domain=[
 ('D01','vpd','일반 지식','6.38;6.131','누적 비율 대신 시간별 현재·1차/2차차분·짧은 창'),
 ('D02','rad_shade','사용자 자료','6.16','커튼 0=닫힘 해석을 적용한 시각별 일사×열림 대리'),
 ('D03','rad_thermal','사용자 자료','6.16','차광과 보온 커튼 경로를 분리하고 1시간 변화를 검증'),
 ('D04','temp_gap','일반 지식','6.16;6.131','내외 온도차의 현재/지연/짧은 창을 따로 검증'),
 ('D05','humidity_gap','일반 지식','6.117','장기 수분수지 대신 1시간 변화/구동기 전환과 결합'),
 ('D06','moisture_flux','일반 지식','6.117;6.118','이전날 연결 제거·동일날 짧은 시간 창과 변화량'),
 ('D07','fan_vpd','일반 지식','6.131','0~6시 선형 진단 대신 매시각·지연 반응/새 기준선'),
 ('D08','heating_vpd','일반 지식','6.131','0~6시 선형 진단 대신 매시각·지연 반응/새 기준선'),
 ('D09','fog_vpd','사용자 자료','6.16','포그 작동 전환 전후 수요 대리의 짧은 반응'),
 ('D10','vent_co2','사용자 자료','6.116','광합성 추정 대신 환기와 CO2의 시간별 동시 변화'),
 ('D11','co2_dose_response','사용자 자료','6.119','공급 중 CO2 1시간 반응을 구동기 전환별 분리'),
 ('D12','co2_uptake_proxy','사용자 자료','6.116','하루 요약 대신 인과적 시간별 마스크·변화율'),
 ('D13','radiation_flow','사용자 자료','6.38;6.175','다일 평균 제외·동일날 1/2/3/4h 변화량/누적'),
 ('D14','temperature_flow','사용자 자료','6.28;6.130','실제 배지온도/정답 보상 없이 실내 입력의 짧은 지연'),
 ('D15','humidity_flow','사용자 자료','6.117','수분수지 결합 없이 습도 자체의 1시간 변화'),
 ('D16','co2_flow','사용자 자료','6.18;6.116','공급/환기 분리 전 기본 CO2 1차·2차차분'),
 ('D17','vent_event','사용자 자료','6.38','개시온 설정 역추정 대신 매시각 switch/run/event-age'),
 ('D18','shade_event','사용자 자료','6.38','설정값 대신 커튼 현재 상태와 전환 후 경과시간'),
 ('D19','thermal_event','사용자 자료','6.38','설정값 대신 보온 커튼 현재 상태와 전환 후 경과시간'),
 ('D20','heating_event','일반 지식','6.38','난방 개시온 대신 매시각 switch/run/event-age'),
 ('D21','fan_event','사용자 자료','6.11;6.131','밀폐 하루 진단 대신 현재 팬 연속상태와 전환'),
 ('D22','fog_event','사용자 자료','6.16','포그 하루 평균 대신 연속작동/전환 후 환경 반응'),
 ('D23','co2_event','사용자 자료','6.119','CO2 공급 하루 평균 대신 연속작동/전환 후 반응'),
 ('D24','sealed_run','일반 지식','6.11;6.113','다일 사슬 제거·동일날 연속 무환기/팬정지 길이'),
]
registry=[]
def add(cid,stage,family,source,previous,difference,status='PENDING',params=None):
    registry.append({'candidate_id':cid,'stage':stage,'family':family,'source':source,
        'previous_catalog':previous,'difference_from_previous':difference,'status':status,
        'parameters':json.dumps(params or {},ensure_ascii=False),'performance':'UNTESTED'})
for cid,fam,src,prev,diff in domain:add(cid,1,fam,src,prev,diff,params={'windows':[1,2,3,4,6],'lags':[1,2,3,4,6]})
for k,(fam,w) in enumerate(itertools.product(['vpd_rad','moisture_flux','rad_shade','temperature_flow'],[1,2,3,4,6,12]),1):
    add('L%02d'%k,2,fam,'논문/인터넷: 출처별 읽기 범위 확인 후 활성화','6.38;6.117;6.130',
        '새 기준선·시각별 지연/변화·각 window를 독립 대조; 1단계와 중복이면 재실행 금지',
        'WAIT_STAGE1_AND_SOURCE_REVIEW',{'window':w})
for i,col in enumerate(RAW,1):add('R%02d'%i,3,'raw_single','대회 원본','6.114;6.362;6.367','선형/원입력 기준 및 지정 기준선에서 단일 열 효과를 별도로 확인','WAIT_STAGE2',{'columns':[col]})
for i,cols in enumerate(itertools.combinations(RAW,2),1):add('P%03d'%i,3,'raw_pair','대회 원본','6.114;6.362;6.367','미사용 원열 쌍을 조합으로 검증','WAIT_STAGE2',{'columns':list(cols)})
for i,col in enumerate(RAW,1):
    add('T%02d'%i,3,'raw_temporal','대회 원본','6.18;6.38;6.368','전체 운영 묶음 대신 원열별 짧은 시간 동역학','WAIT_STAGE2',{'columns':[col]})
    add('A%02d'%i,3,'raw_ablation','대회 원본','6.362;6.367','원열 단위 재학습 제거와 간접 경로 제거를 구분','WAIT_STAGE2',{'columns':[col]})
add('ALL14',3,'all_raw','대회 원본','6.114','10열 본체와 외기4 포함 14열 본체의 동일 검증 재학습','WAIT_STAGE2')
with (HERE/'feature_candidates_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(registry[0]));w.writeheader();w.writerows(registry)

tm_path=ROOT/'집/클로드/research/local/tm1_set_v1.csv'
tm={(r['farm'],int(r['day'])) for r in read(tm_path)}
assert len(tm)==111
wt1_path=ROOT/'집/클로드/research/local/ec3_WT1_all.csv'
wt2_path=ROOT/'집/클로드/research/local/ec3_WT2_all.csv'
o=read(wt1_path);o2=read(wt2_path)
folds={}
for r in o:folds.setdefault((r['validator'],int(r['validation_fold'])),set()).add(r['row_id'])
write('fold_registry_v1.json',{'source_SHA256':sha(wt1_path),'TM_SHA256':sha(tm_path),
    'TM_days':sorted(tm),'folds':[{'validator':v,'fold':i,'query_ids':sorted(ids)} for (v,i),ids in sorted(folds.items())],
    'training_policy':'holdout与locked farm-day±1 purge; training/anchor/context IDs须逐fold重新记录',
    'confirmation':'保留未使用新seed及新fold布局; 单次判定, 未运行'})
scores={}
for validator in ['DIAG10','P2LOO','EL1']:
    rows=[r for r in o2 if r['validator']==validator]
    if validator=='DIAG10':assert all((r['farm'],int(r['day'])) in tm for r in rows)
    days={}
    for r in rows:days.setdefault((r['farm'],int(r['day'])),[]).append(float(r['sub_ec']))
    means={key:math.fsum(vals)/len(vals) for key,vals in days.items()}
    score={'rows':len(rows),'days':len(days),'by_seed':{}}
    for s in SEEDS:
        errors=[(float(r['sg4_%d'%s])-float(r['sub_ec']))**2 for r in rows]
        a=math.sqrt(math.fsum(errors)/len(errors));b=math.sqrt(sum(errors)/len(errors))
        assert abs(a-b)<1e-12
        segments={}
        for name,high in [('normal',False),('high',True)]:
            ee=[e for e,r in zip(errors,rows) if (means[(r['farm'],int(r['day']))]>=1)==high]
            segments[name]={'rows':len(ee),'RMSE':math.sqrt(math.fsum(ee)/len(ee))}
        score['by_seed'][str(s)]={'RMSE':a,'segments':segments}
    scores['TM' if validator=='DIAG10' else validator]=score
write('historical_baseline_recheck_v1.json',{'source_SHA256':sha(wt2_path),'configuration':'0.6 R3_DP1 + 0.4 PFN + SG2',
    'provenance':'Claude WT2 sg4 saved OOF. 独立算术复核不是clean retraining或因果审计通过',
    'scores':scores,'training_performed':False})
registration={
    'primary_metric':'RMSE','baseline':'season_v2 + DP1 + PFN .4 + SG2',
    'baseline_status':'历史OOF已重新算术核对; exact CPU复现和完整因果审计尚未通过',
    'exploration_seeds':SEEDS,'validators':['TM111 on DIAG10 layout','P2LOO','EL1'],
    'stage_candidate_caps':{'domain':24,'literature':24,'data_driven_initial_batch':256},
    'initial_registry_count':len(registry),'raw_subset_universe':2**14-1,
    'coverage':'此批不是数学上所有变换的全集; 后续分批登记所有16383非空原列子集和family组合',
    'confirmation':{'new_seeds':'尚未使用seed须与全历史ledger比对后封存','new_folds':'尚未使用布局须历史审计后封存',
        'one_run_only':True,'all_seed_validator_cells_better':True,'p_worse':'TM seedmean paired farm×5-day block bootstrap < .025 / total finalist count',
        'bootstrap_draws':200000,'alpha':0.025,'selection':'所有筛选次数/重复使用验证标签写入ledger; 新布局确认不得拿原探索p值代替'},
    'causality':'same farm 0..h query prefix, public training stores isolated, fit training only, no test adaptation',
    'temperature':'observed current evaluation temp unavailable; aux/anchor/crossfit OOF candidates only; outer+inner hide both labels',
    'neighbours':'PF1/PF2由Claude进行, 本次不启动GPU/重复邻居训练; 接收文件只读独立审计',
    'restart':'reuse only matching input/code/env/fold/training/anchor/context/postprocess/seed hashes; atomic receipt after prediction verification',
    'checkpoint':'RUNNING必须核对PID+启动时间+command/tool handle; 单纯文件存在不是完成',
    'baseline_gate':'禁止在exact baseline与future perturbation审计未通过时报告candidate改善',
}
write('preregistration_v1.json',registration)
plan='''# EC 基础重检计划 v1（2026-10-07， 집 코덱스）

用户指定基准: 계절 v2 + DP1 + TabPFN .4 + SG2。与submission14 .2版本分开。
原资料:14原输入、row_id、同温室输入prefix、公开EC/温度正答、公开全部train资料的辅助机会。
外部数值资料禁用。领域资料来源和论文来源逐项标明，模型效果只靠新实验。

第一阶段24候选上限，第二阶段24，第三阶段每批256。先登记具体候选。
每个已有失败family必须引用catalog和区别。没有区别的重复项不启动。
短时:同日1/2/3/4/6h lag/window、1次/2次差分、开关、run、事件后时间；12/24h补充。
日内形状只能0..h prefix。前日对照须单独连接可信度及abstain审计，不能按row_id连续假定。
一阶段结束才开启二阶段，二阶段结束才开启三阶段；每阶段独立严厉critic review。

TM111固定、P2LOO、EL1，探索3seed。constant和raw-only CPU模型是基础对照，不能替代指定baseline。
历史sg4 OOF只是算术参考。exact baseline重现+全pipeline未来输入/别温室/顺序/单query不变性审计是正式改善比较的关卡。
精确baseline CPU实现或原缓存因果问题没解决时，候选只做实现/数据诊断，不冒称通过。
外层validation及inner query的EC和temp必须同时从学习/anchor/context移除。
TM由历史全日test相似性选出，只作为用户指定固定诊断，不给特征fit或系数调节提供test统计。
fresh seeds+fresh fold layout最终一次确认，提前封存、核对历史使用情况。所有seed×3validator改善并TM P(worse)<.025/k。
多重比较台账包括所有被筛选和重试的variant，不能只记录winner；最终k定义所有进入一次确认的候选。
排名:指定baseline retrain增量/移除ΔRMSE、seed×validator稳定、normal/highEC、hour/farm/pass、block CI。
raw14非空子集16383个，只是有限原列空间；派生变换无限，registry逐批扩展并显示覆盖分母，不能冒称完整。
独立critic意见保存issue ID→fix→retest证据。checkpoint严格hash、原子receipt、保存预测重算、PID真实存活核对。
'''
assert not(HERE/'PLAN_v1.md').exists()
(HERE/'PLAN_v1.md').write_text(plan,encoding='utf-8')
(HERE/'PROGRESS.md').write_text('''# EC 변수 전수 재점검 — 재개 지점

- 목표 미완료. 아직 신규 후보 학습/채택/제출 없음.
- 완료: 원자료·카탈로그 스냅샷, 후보 등록, TM/기존 fold ID 보존, 역사 기준선 산술 재검산.
- 다음: 계획 독립 비평 반영 → 한국어 계획 v2 → 시각별 특징 구현/인과 검사 → exact baseline CPU 재현 경로 감사 → ① 도메인 후보 실행.
- CPU만 사용. PF1/PF2 GPU와 이웃 특징은 클로드 진행을 존중하고 결과 감사 후 반영.
- baseline SG2 .4 역사 OOF 재채점은 재학습 및 규정 적합성 인증이 아님.
- 재시작: 이 파일 → preregistration_v1.json → feature_candidates_v1.csv → 체크포인트 receipt를 읽고 이어감.
- 원파일은 보존. 이 PROGRESS.md만 사용자 요청에 따라 단계별 갱신.
''',encoding='utf-8')
print(json.dumps({'source_rows':len(x),'counts':counts,'candidate_counts':{str(s):sum(r['stage']==s for r in registry) for s in [1,2,3]},'catalog_records':len(records),'scores':scores},ensure_ascii=False))
