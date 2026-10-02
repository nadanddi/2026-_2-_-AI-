"""완료된 실험 결과만 다시 읽는 독립 집계. 학습/제출 예측 없음."""
from pathlib import Path
import csv, json, math, hashlib
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ANALYSIS = ROOT / '집/코덱스/analysis'
LOCAL = ROOT / '집/코덱스/local'
ARMS = ['drop_et', 'drop_lgb', 'drop_mlp', 'drop_pfn', 'catboost']
NAMES = {'v2':'기존 v2', 'drop_et':'ET 제거', 'drop_lgb':'LGB 제거',
         'drop_mlp':'MLP 제거', 'drop_pfn':'TabPFN 제거', 'catboost':'CatBoost 단독'}
VALIDATORS = ['DIAG10', 'A', 'B', 'EXT10', 'EXT12']
SEEDS = [7,101,2024]
def read_json(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def rmse(p,y): return math.sqrt(math.fsum((a-b)**2 for a,b in zip(p,y))/len(y))
def calculate(rows,arms):
    groups = {}
    for r in rows:
        key = (r['validator'],int(r['validation_fold']),r['row_id'])
        groups.setdefault(key,[]).append(r)
    out = []
    for validator in VALIDATORS:
        items = [rs for k,rs in groups.items() if k[0]==validator]
        assert all(len(rs)==3 and {int(r['seed']) for r in rs}==set(SEEDS) for rs in items)
        y = [float(rs[0]['sub_ec']) for rs in items]
        for arm in ['v2']+arms:
            p = [math.fsum(float(r[arm]) for r in rs)/3 for rs in items]
            out.append(dict(validator=validator,arm=arm,rmse=rmse(p,y),n=len(y)))
    return out
def verify_numeric(rows,evaluation,arms):
    maximum = 0.
    for record in evaluation['scores']:
        g = [r for r in rows if r['validator']==record['validator'] and int(r['seed'])==record['seed']]
        p = [float(r[record['arm']]) for r in g]
        y = [float(r['sub_ec']) for r in g]
        assert len(g)==record['n']
        maximum=max(maximum,abs(rmse(p,y)-record['rmse']))
    assert maximum<1e-12
    for arm in arms:
        scores=[r for r in evaluation['scores'] if r['arm']==arm]
        boots=[r for r in evaluation['bootstrap'] if r['arm']==arm]
        assert len(scores)==15 and len(boots)==3
        decision=next(r for r in evaluation['decisions'] if r['arm']==arm)
        direction=all(r['delta_rmse']<0 for r in scores)
        statistical=all(r['p_worse']<.005 and r['ci_mse_high']<0 for r in boots)
        assert decision['adopted']==(direction and statistical)
    return maximum
def main():
    member=ANALYSIS/'ec_member_ablation_20261002_v1'
    cat=ANALYSIS/'ec_catboost_20261002_v1'
    assert read_json(member/'completion.json')['status']=='COMPLETE'
    assert read_json(cat/'completion.json')['status']=='COMPLETE'
    me=read_json(member/'evaluation.json')
    ce=read_json(cat/'common_evaluation.json')
    paths=[LOCAL/'ec_member_ablation_20261002_v1/oof_predictions.csv', LOCAL/'ec_catboost_20261002_v1/oof_predictions.csv']
    mr,cr=[csv_read(p) for p in paths]
    assert len(mr)==len(cr)==83160
    max_member=verify_numeric(mr,me,ARMS[:4])
    max_cat=verify_numeric(cr,ce,['catboost'])
    aggregate=calculate(mr,ARMS[:4])+[r for r in calculate(cr,['catboost']) if r['arm']=='catboost']
    base_member={(r['validator'],r['validation_fold'],r['seed'],r['row_id']):float(r['v2']) for r in mr}
    base_gap=max(abs(float(r['v2'])-base_member[(r['validator'],r['validation_fold'],r['seed'],r['row_id'])]) for r in cr)
    assert base_gap<1e-12
    assert read_json(cat/'completion.json')['oof_sha256']==sha(paths[1])
    decisions=me['decisions']+ce['decisions']
    assert len(decisions)==5
    selected=[d['arm'] for d in decisions if d['adopted']]
    r3audit=read_json(member/'reconstruction_audit.json')
    assert len(r3audit)==66
    r3_max=max(d['refit_finished_r3_max_difference'] for d in r3audit)
    pfn_max=max(d['pfn_inverse_shrink_roundtrip_max_difference'] for d in r3audit)
    lookup={(r['arm'],r['validator']):r['rmse'] for r in aggregate}
    table=['| 비교 | DIAG10 | A | B | EXT10 | EXT12 | 판정 |','|---|---:|---:|---:|---:|---:|---|']
    for arm in ['v2']+ARMS:
        status='기준' if arm=='v2' else ('공개 기준 통과·추가 확인 필요' if arm in selected else '기각')
        table.append('| '+NAMES[arm]+' | '+' | '.join(f'{lookup[(arm,v)]:.6f}' for v in VALIDATORS)+' | '+status+' |')
    gate=['| 비교 | 개선한 시드×검증기 | DIAG10 p_worse 범위 | 99% CI 통과 시드 |','|---|---:|---:|---:|']
    for d in decisions:
        b=[r for r in me['bootstrap']+ce['bootstrap'] if r['arm']==d['arm']]
        gate.append(f"| {NAMES[d['arm']]} | {d['improving_seed_validator_cells']}/15 | {min(r['p_worse'] for r in b):.5f}~{max(r['p_worse'] for r in b):.5f} | {sum(r['ci_mse_high']<0 and r['p_worse']<.005 for r in b)}/3 |")
    conclusion=(f'5안 중 공개 선별 기준 통과 {len(selected)}안. 최종 잠금 및 추가 독립 확인 전에는 채택을 확정하지 않는다.' if selected else '5안 모두 사전 채택 기준 불합격. 기존 EC 후보 v2를 유지하며 새 후보는 없다.')
    text=f'''# EC 모델 비교 결과 · 2026-10-02 집 코덱스

{conclusion}

기존 v2 구성원 하나씩 제거4안과 CatBoost 단독1안을 실행했다. 각 안 22분할×3시드, 비잠금360일8640행(DIAG10), 전체검증 발생행27,720×3시드=83,160행이다. 표는 세 시드의 예측 평균 RMSE이며 공식 점수나 독립 홀드아웃 점수가 아니다. 사전등록 main 672addc, 고정 캐시 워커 cbd5839.

{chr(10).join(table)}

판정은 시드 평균이 아니라 각 시드×검증기15칸 전부 개선과 각 시드 DIAG10 농장층화5일블록20,000회에서 p_worse<.005, 99% CI 상한<0이다. 동시5안 본페로니를 적용했고 점수를 보고 기준이나 비중을 바꾸지 않았다.

{chr(10).join(gate)}

## 검산과 범위

- csv+math.fsum 독립 산술 vs 실행 NumPy RMSE 최대차: 멤버 {max_member:.3g}, CatBoost {max_cat:.3g}; 두 실험 기준v2 차이 {base_gap:.3g}.
- 66개 재학습 R3와 기준캐시 최대차 {r3_max:.3g}; PFN복원 왕복 최대차 {pfn_max:.3g}. 22원캐시 범위 자름0건, 독립 행렬 역산과 시드간PFN일치 확인.
- 공통 평가에 동일후보·누락시드/분할/행·중복행 반례 검사를 수행했다. 실제 결과의 라벨/후처리/해시/부트스트랩 독립 감사는 각 모델 폴더와 ec_member_audit_20261002_v1에 있다.
- 특징은 MASK허용 입력과 기존 인과 특징만, 전처리는 fit 안에서 학습. 잠금40일은 숫자변환 전 제외하고 검증일·잠금일 ±1일 purge. test_X 값/미래/타온실 입력/외부데이터/리더보드 비중선택/제출물 생성0.

## 해석과 한계

기각은 시험한 제거비중과 고정 CatBoost 설정에 대한 결과다. 다른 알고리즘이나 설정 전체가 쓸모없다는 증명은 아니다. 검증기는 반복 사용했고 PFN 묶음은 기존4문맥 한 묶음이다. 최종 잠금40일을 채점하지 않았으므로 비공개 성능 향상은 주장하지 않는다. 시드평균 수치는 진단용이며 개별시드의 실패를 대신하지 않는다. 라벨로 나눈 세그먼트와 하루수준/모양 분해는 사후 오차 진단이며 평가 특징이나 보정비중 선택에 쓰지 않는다. 독립 재계산과 전체분할 기준 기각 판정의 신뢰도는 높고, 공식 점수 전이 추정은 하지 않는다.
'''
    (HERE/'결과보고서.md').write_text(text,encoding='utf-8')
    report={'status':'PASS','selected_public_arms':selected,'decisions':decisions,'ensemble':aggregate,
            'cross_checks':{'member_rmse_max_error':max_member,'catboost_rmse_max_error':max_cat,'baseline_max_error':base_gap,
                            'refit_r3_max_error':r3_max,'pfn_roundtrip_max_error':pfn_max},
            'oof_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
            'final_lock_scored':False,'submission_created':False}
    (HERE/'verification_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
