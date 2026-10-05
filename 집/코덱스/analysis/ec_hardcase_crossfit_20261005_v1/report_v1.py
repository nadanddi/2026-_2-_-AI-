from pathlib import Path
import json
H=Path(__file__).resolve().parent
def main():
    v=json.loads((H/'verification_v5.json').read_text(encoding='utf-8'));f=json.loads((H/'fresh_numbers_v3.json').read_text(encoding='utf-8'));assert v['status']=='PASS_NESTED_OOF_LEARNING_REPLAY' and f['status']=='PASS_STDLIB_COUNTS_RATIOS_SUPPORT_BOOTSTRAP'
    rows=[r for r in f['results'] if r['v']=='DIAG10' and r['hour']==23 and r['selection']=='GUARD'];get=lambda mode:[r for r in rows if r['mode']==mode]
    def ran(z,key,group='model'):
        vals=[r[group][key] for r in z];a,b=min(vals),max(vals);return str(a) if a==b else f'{a}~{b}'
    support=f['support'];joint=sum(r['hard_high']>=3 and r['hard_low']>=3 for r in support)
    lines=['# 경계 사례 공동 학습 결과','',
        '2026-10-06 집 코덱스(10-05 시작한 실험 폴더 유지). 학습 사례를 넓혀 낮게 예측된 진짜 고EC와 일반날 오탐을 함께 학습했다. 고EC 포착 유지와 오탐 감소를 동시에 달성하지 못해 채택하지 않는다.','',
        '사전94c2ba9. 기존 외부 DIAG10/A/B×3seed, 외부 학습 날짜 전체에 4fold 5기록일 블록 OOF와 ±1일 purge. 평가 날짜는 어느 대리모델·참조·분류기 학습에도 포함하지 않았다. 기존 공개 OOF를 다른 외부fold에서 가져와 합치는 방법은 쓰지 않았다. 최종 검산 대상은 대리EC모델300개와 LR구별기120개, train/outerCSV240개다.','',
        '경계 사례 정의용 EC는 기존 R3의 LightGBM tweedie 구성만 새로 학습한 대리 예측이다. 완전한 기존 A(R3/TabPFN 혼합) 재현이 아니다. 기존 실제 계절v2 A가 외부 비교 기준이며 GUARD의 첫 판정이다. 동일 데이터22특징/LR C1/문턱.5에서 UNIFORM은 class균형 가중치, HARD는 진짜고EC & 해당시각 대리prefix<1.2 또는 일반날 & 대리prefix≥.9인 행에 4배 가중치. 평균가중치1로 정규화했다. 외부정답을 보고 문턱이나 가중치를 바꾸지 않았다.','',
        '하루 끝23시 DIAG10 외부360온실별기록일(고EC31/일반329), 세시드7/101/2024:','',
        '| 방식 | 고EC 포착/31 | 일반날 오탐/329 |','|---|---:|---:|',
        f"| 기존 A |{ran(get('UNIFORM'),'tp','baseline')}|{ran(get('UNIFORM'),'fp','baseline')}|",
        f"| UNIFORM + GUARD |{ran(get('UNIFORM'),'tp')}|{ran(get('UNIFORM'),'fp')}|",
        f"| HARD + GUARD |{ran(get('HARD'),'tp')}|{ran(get('HARD'),'fp')}|",'',
        'GUARD는 기존A prefix≥.9가 잡은 날 중 분류점수≥.5인 날만 남긴다. 오탐 비증가는 구조적으로 보장되므로 그 자체를 학습 효과로 해석하지 않는다. DIRECT는 새분류기만으로 판정한다. HARD DIRECT는고EC27~29포착/오탐41~44로 기존대비 포착 감소·오탐 증가다. GUARD의 하루끝 결과를0시 성능으로 해석하지 않는다.','',
        '세시드별 DIAG23시 상세:','',
        '| seed | UNIFORM TP/FP | HARD TP/FP | 기존 TP/FP |','|---|---:|---:|---:|']
    for s in [7,101,2024]:
        a=next(r for r in get('UNIFORM') if r['seed']==s);b=next(r for r in get('HARD') if r['seed']==s)
        lines.append(f"|{s}|{a['model']['tp']}/{a['model']['fp']}|{b['model']['tp']}/{b['model']['fp']}|{a['baseline']['tp']}/{a['baseline']['fp']}|")
    lines+=['','外23시 A/B는 원검증기의 중복 날짜 출현단위이며 독립날짜수와 다르다:','',
        '| 검증기·seed | UNIFORM TP/FP | HARD TP/FP | 기존 TP/FP | 표본 출현수/고EC출현수 |','|---|---:|---:|---:|---:|']
    for validator in ['A','B']:
        for s in [7,101,2024]:
            a=next(r for r in f['results'] if r['v']==validator and r['seed']==s and r['mode']=='UNIFORM' and r['hour']==23 and r['selection']=='GUARD');b=next(r for r in f['results'] if r['v']==validator and r['seed']==s and r['mode']=='HARD' and r['hour']==23 and r['selection']=='GUARD')
            lines.append(f"|{validator}·{s}|{a['model']['tp']}/{a['model']['fp']}|{b['model']['tp']}/{b['model']['fp']}|{a['baseline']['tp']}/{a['baseline']['fp']}|{a['model']['n']}/{a['model']['high']}|")
    lines+=['',f"사전GUARD 관문 결과: {json.dumps(v['primary'],ensure_ascii=False)}. 모든seed×검증기 TP감소0/FP증가0 및 DIAG매seed FP엄격감소를 요구했다. 두안 모두 불통과. 결합RMSE나 기존의 RMSE통계채택조건을 통과했다고 주장하지 않는다.",'',
        f"공동학습 표본(각외부tr/각seed): {min(r['n'] for r in support)}~{max(r['n'] for r in support)}일, 고EC {min(r['high'] for r in support)}~{max(r['high'] for r in support)}일. 낮은대리예측 진짜고EC {min(r['hard_high'] for r in support)}~{max(r['hard_high'] for r in support)}일, 일반날대리오탐 {min(r['hard_low'] for r in support)}~{max(r['hard_low'] for r in support)}일. 두종류 각3일이상은{joint}/{len(support)}구간;일부는일반오탐1일뿐이므로모든구간의학습정보가충분했다고할수없다. 내부proxy훈련고EC는최소4일로0모델문제는없었다.",'',
        '대조 해석: HARD는 UNIFORM보다 같은 외부DIAG에서 고EC를3~4일 더 포착했다. 가중치 하나를 바꾼 효과는 관찰됐지만, 기존A보다2~3일 더 놓치면서 오탐2~3일을 줄였다. 고EC포착 유지라는 사전 목표를 만족하지 못했다. 학습 범위 넓힘과 이전모델 대비 순수효과는 대리모델 변경도 함께 있어 분리하지 못한다. 이번 실패가 모든구별기/입력정보가치 부재의 증거는 아니다.','',
        '학습·외부 차이: 구별기의 train 점수는 OOF 대리피처에 분류기를 적합한 resubstitution 점수이며 별도OOF 분류성능이 아니다. train/outer DIRECT 상세는 verification_v5.results에 보존했다. GUARD train의 첫예측은 대리,outer의첫예측은 실제A이므로 이 두GUARD간 차이를 같은게이트의순수과적합격차로 해석하지 않는다. 가중치학습 LR점수는 실제고EC확률로교정되지 않았다.','',
        f"검산: scalar LR 최대오차{v['max_error']:.3g}, Tweedie dumped-tree scalar오차{v['proxy_scalar_error']:.3g}, 학습/scaler gradient최대{v['max_gradient']:.3g}, 새재학습 재현{v['repeats']}개(2대리/2LR), 모델/피처/라벨·weight·참조ID/날짜purge/확률·선택/原CSV counts·비율 및 rank/sklearn vs 쌍AUC교차검산PASS. 数値根拠 verification_v5.json/fresh_numbers_v3.json, 실행 verify_v5.py/fresh_numbers_v3.py. baseline의ECprefix를 Brier확률교정 지표로 해석하지 않았다.",'',
        '블록진단: DIAG 온실별기록순서5일block,3seed평균·10k재표본. 일반오탐차와전체오분류차의CI를새로계산했다. GUARD 구조상오탐증가는없으므로오탐CI만으로모델채택하면안된다. 비공개/리더보드로가중치·계수를계산하지 않았다.','',
        '```json',json.dumps(f['bootstrap'],ensure_ascii=False,indent=2),'```','',
        '검산·실행 기록: 순차실행10개완료구간은불변재사용,미완료구간과그산출물은보존하고새suffix파일로3process 재개(65f6cd8). 같은원모델/피처/가중치/문턱. 검산helper이름은학습전새버전수정,추가fullproxy재학습도별도고정했다. 검산v4의부분12시간표본이일별prefix조건을만족못한오류는새v5에서24시간으로교정;원로그보존/모델·기준변경0. 검산v5 초기 shell인용 오류로파일미생성한로그도보존했고실제성공로그는verify_v5_ready.log다.','',
        '비평과 다음 관문:','',
        '- 공동학습에 양쪽사례가 포함됐는가? 예(신뢰도높음),최종OOF날짜count재계산으로확인. 다만3구간은일반오탐<3일이며대리기준으로정의한어려움이다.',
        '- 목표를 달성했는가? 아니오(신뢰도높음),수치교차검산/3seed×3검증기고정비교에서실패. 공개날짜반복탐색이라신규독립확증아님.',
        '- 특정입력이 부족하다는 결론인가? 미확정. 표본·대리예측과실제A경계의차이·22열LR표현력·고정문턱이함께남는다. 고EC분류전략전체의불가능성을주장하지않는다.',
        '- 다음은 같은A의독립교차예측으로경계정의를맞추고,놓친고EC와제거한일반날의특징·출처차이를대조하는단계다. 현재외부날짜의정답을보고.5문턱이나4배가중치를바꾸지않았다. 전문가크기모델/결합RMSE는별도검증필요.',
        '', '기존A 유지/채택·확정구성·제출0,SG2최신합본효과미시험/EL1잠금미채점/원운영위PDF미확보라는범위한계유지. Claude의자정연결6.345는중복실행하지않았고TabDPT기존worker변경0. 최종검증대상CSV·모델은코덱스local폴더의Drive동기화대상.']
    with (H/'실험결과_v1.md').open('x',encoding='utf-8') as out:out.write('\n'.join(lines)+'\n')
    print('REPORT_WRITTEN')
if __name__=='__main__':main()
