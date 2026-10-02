"""Fresh numerical and adversarial review; no fitting or selection."""
from support import *
import math

def main():
    proofs = ['verification.json', 'extra_verification.json',
              'bootstrap_verification.json', 'conditional_verification.json',
              'cpu_replay.json', 'gap_verification.json', 'oracle_verification.json']
    assert all(json.loads((HERE/p).read_text(encoding='utf-8'))['status']=='PASS' for p in proofs)
    main_result = json.loads((HERE/'result.json').read_text(encoding='utf-8'))
    extra_result = json.loads((HERE/'extra_result.json').read_text(encoding='utf-8'))
    data = pd.concat([pd.read_csv(OUT/'oof.csv'), pd.read_csv(OUT/'extra_oof.csv')])
    scores = main_result['scores'] + extra_result['scores']
    decisions = main_result['decisions'] | extra_result['decisions']
    for s in scores:
        d = data[(data.arm==s['arm']) & (data.validator==s['validator']) &
                 (data.seed==s['seed']) & (data.context==s['context'])]
        a = math.sqrt(math.fsum((float(r.baseline)-float(r.y))**2 for r in d.itertuples())/len(d))
        b = math.sqrt(math.fsum((float(r.candidate)-float(r.y))**2 for r in d.itertuples())/len(d))
        assert abs(a-s['baseline_rmse'])<1e-12 and abs(b-s['candidate_rmse'])<1e-12
    fitting = {}
    for p in OUT.glob('*_meta.npz'):
        z = dict(np.load(p))
        choices = [('TBIAS','cal_z','bias_coef'),('TGATE','cal_design','gate_coef')] if p.name.startswith('T_') else [('ESTATE','cal_design','coef')]
        for arm, design, coef in choices:
            y=z['cal_target']; w=z.get('cal_w',np.ones(len(y))); after=y-z[design]@z[coef]
            a=math.fsum(float(ww)*float(v)**2 for ww,v in zip(w,y))
            b=math.fsum(float(ww)*float(v)**2 for ww,v in zip(w,after))
            fitting.setdefault(arm,[]).append(100*(math.sqrt(b/a)-1))
    for p in OUT.glob('*_extra.npz'):
        z=dict(np.load(p)); y=z['cal_target']; w=z.get('cal_w',np.ones(len(y)))
        after=y-z['cal_design']@z['coef']
        a=math.fsum(float(ww)*float(v)**2 for ww,v in zip(w,y))
        b=math.fsum(float(ww)*float(v)**2 for ww,v in zip(w,after))
        fitting.setdefault('TBIAS_PH' if p.name.startswith('T_') else 'ESTATE_PAR',[]).append(100*(math.sqrt(b/a)-1))
    lines=['# 최종 반론 검토와 신뢰도','',
           '66점수 셀을 최종 CSV에서 math.fsum으로 다시 계산했다. 원 검산과 RMSE 차 <1e-12. 모든 내부/외부 정답 분리, 과거 source, 계수, bootstrap 및 첫 전문가 재학습 증거는 7개 PASS 산출물에 보존했다.','',
           '| 구현 | 보정기 적합 잔차 RMSE 변화 범위 | 최종 판정 |','|---|---:|---|']
    for arm in ['TBIAS','TBIAS_PH','TGATE','ESTATE','ESTATE_PAR']:
        q=[s for s in scores if s['arm']==arm]
        passed=all(s['delta_pct']<0 for s in q) and all(s['p_worse']<ALPHA and s['ci_mse'][1]<0 for s in q if s['validator']=='DIAG10')
        expected=('PUBLIC_PASS_NO_NEW_LOCK' if arm.startswith('ESTATE') else 'PASS_REQUIRES_GUARD_EL1') if passed else 'REJECT'
        assert decisions[arm]==expected
        v=fitting[arm]
        lines.append(f'| {arm} | {min(v):+.4f}% ~ {max(v):+.4f}% ({len(v)}개 적합) | {expected} |')
    lines += ['',
        '적합 잔차 수치는 보정기를 학습한 자료에서의 값이며, clip·저온 gate·비중 제약 투영 전 선형 적합값이다. 외부 예측 점수와 같은 통계량이 아니고 성능 증거로 사용하지 않는다. 내부 적합과 외부 검증의 차이는 과적합뿐 아니라 내부/외부 전문가 학습량 차이와 구간 분포 이동으로도 생긴다.','',
        '- 주장: 다섯 구현은 사전 채택 기준 미충족. 신뢰도 높음(이번 고정 검증에 한정). 반론: 일부 검증에서는 개선했다. 답: 전체 방향과 DIAG 유의성 조건을 함께 요구했으므로 일부 개선으로 교체하지 않는다. 모든 변형의 가능성을 기각한 결과는 아니다.',
        '- 주장: TGATE의 DIAG 개선이 새 구간까지 일반화되지 않았다. 신뢰도 높음(관측한 EXT10/12). 반론: 수축 강도를 바꾸면 좋아질 수 있다. 답: 가능하지만 이번 점수로 설정을 다시 고르지 않았다. 현재 조건 특징과 고정 제약의 구현에 한정한다.',
        '- 주장: EC 과거 상태는 안정적인 추가 정보를 제공한다는 근거가 부족하다. 신뢰도 중간. 반론: DIAG가 개선됐고 조건부 진단 일부도 개선했다. 답: B 악화·조건부 진단 54/132개 개선이며, 중첩 표본과 정규화 변화도 영향을 준다. 정보 부재나 인과관계를 입증한 것은 아니다.',
        '- 주장: 평가 구조에 맞는 관측 간격을 고려했다. 신뢰도 높음(metadata). 반론: label source 가용성이 실사용과 다를 수 있다. 답: 공개360일과 실제 기준306일을 따로 기록했다. 5일 추가 지연 stress는 보정기만 변경하며 전체 전문가 재학습이나 새로운 독립 검증기를 대신하지 않는다.',
        '- 미사용 최종 홀드아웃으로 최종 모델을 채택하지 않았다. EC 잠금 재열람·재채점 없음. 반복 CV, CODEX 시드의 실질 중복, 날씨 그룹 공유, 원TF 전역 입력 rank 가중치가 남아 있으며 독립적인 미래 리더보드 개선은 주장하지 않는다.',
        '- 최초 EC EXT12 내부 정의역 실패는 날짜만으로 보완하고 버전·사전 커밋을 남겼다. TF32 오차 초과는 미적용. 보완2안은 초기 한 분할 smoke 확인 후 등록됐으므로 완전 비적응 탐색이라고 부르지 않는다.',
        '- 기존 TK2 계절 실험은 별도 GPU 작업으로 재개됐으며 이 보고서의 완료 범위에 포함하지 않는다.',
    ]
    savej(HERE/'final_review.json',dict(status='PASS',fresh_score_cells=len(scores),decisions=decisions,fit_residual_delta_ranges={k:[min(v),max(v),len(v)] for k,v in fitting.items()},oof_hash=sha(OUT/'oof.csv'),extra_oof_hash=sha(OUT/'extra_oof.csv')))
    (HERE/'최종_반론검토_v1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FINAL_REVIEW_PASS',flush=True)

if __name__=='__main__':
    main()
