import json
import pandas as pd
from run import HERE,OUT

def main():
    r=json.loads((HERE/'result_v2.json').read_text(encoding='utf-8'))
    v=json.loads((HERE/'verification_v2.json').read_text(encoding='utf-8'))
    assert v['status']=='PASS'
    s=pd.read_csv(HERE/'scores_v2.csv');dec=pd.read_csv(HERE/'decisions_v2.csv');risk=pd.read_csv(HERE/'risk_scores_v2.csv');seg=pd.read_csv(HERE/'segments_v2.csv')
    lines=['# 중요도 순 실험 결과 — 2026-10-03 집 코덱스','',
      '사용자 요청으로 9개 항목을 중요도 순으로 실행했다. 최신 W40G 기준선 진단, 모델 비교 10안, 두 목표의 실패위험 진단을 수행했다. 별도 전문가 항목은 하루모델·온도차 모델에 포함해 중복 계산하지 않았다. 실험 결과에 따라 혼합비나 판정 기준을 바꾸지 않았다.', '',
      f"사전 고정 a45ef88, ET 피처순서 정정 e9f5480. 모델 비교 {r['score_cells']}셀, family_k=10, DIAG10 block-bootstrap 20,000회·p_worse<.0025 및 CI 상한<0, 모든 시드/검증기/문맥 방향 개선이 기준이다. EC 3시드×5검증기, 온도 2시드×2문맥×3검증기를 사용했다. EL1은 공개 기준 통과 시 추가 검증할 조건이며, 통과하지 않은 안에는 잠금/EL1을 소비하지 않았다.", '',
      '## 모델별 판정','',
      '| 실험 | DIAG10 변화 범위 | 전체 검증 변화 범위 | 판정 |','|---|---|---|---|']
    for arm in dec.arm:
        a=s[s.arm==arm];d=a[a.validator=='DIAG10'];status=dec[dec.arm==arm].status.iloc[0]
        lines.append(f'| {arm} | {d.delta_pct.min():+.3f}% ~ {d.delta_pct.max():+.3f}% | {a.delta_pct.min():+.3f}% ~ {a.delta_pct.max():+.3f}% | {status} |')
    lines+=['','음수는 RMSE 개선이다. 전체 범위는 서로 다른 검증기의 성능을 평균한 값이 아니다. 상세 분모·폴드 평균/표준편차·시드·문맥·CI는 scores_v2.csv, 목표/농장/구간·고EC 편향은 segments_v2.csv에 있다.','',
      '## 최신 온도 기준선','',
      f"기존 W30G 실패 {r['W40G_old_failure_days']}일 중 W40G에서도 RMSE>.5인 날은 {r['W40G_old_failures_still_fail']}일이다. 그 집합의 pooled RMSE는 {r['W40G_old_failures_old_rmse']:.6f}→{r['W40G_old_failures_new_rmse']:.6f}℃. 원 기준은 입력24시간 완전관측·일평균 |배지−실내|≥2℃. W40G_daily_audit.csv에 400일의 변화와 분모를 보존했다.",
      '이 W40G는 저장 멤버의 시드별 재구성이다. 실제 제출은 BASE 3시드 평균이므로 실제 제출 파일의 60일 점수를 재계산한 것과 다르다. 기존 온도 게이트의 결측 처리도 유지했다.','',
      '## 실패 위험 진단','',
      '| 목표 | 시각 | 일수/실패일수 | AUC | Brier / 상수 Brier | 낮은/높은 위험군 RMSE |','|---|---|---|---|---|---|']
    for row in risk[risk.validator=='DIAG10'].itertuples():
        lines.append(f'| {row.target} | {row.hour} | {row.days}/{row.failure_days} | {row.auc:.4f} | {row.brier:.4f}/{row.constant_brier:.4f} | {row.low_risk_rmse:.4f}/{row.high_risk_rmse:.4f} |')
    comparator=pd.read_csv(HERE/'risk_simple_comparator_v2.csv')
    assert json.loads((HERE/'risk_comparator_verification_v1.json').read_text(encoding='utf-8'))['status']=='PASS'
    lines+=['','AUC는 순위 구별, Brier는 확률 오차다. 상수 Brier는 검증 집합의 실제 비율을 아는 사후 참고값이므로 배포 가능한 모델의 점수로 취급하지 않는다. 위험군은 시각별 예측확률 하위/상위 25%이고 동률을 포함한다. 위험 예측에 현재/미래 정답을 입력하지 않았다. 기존 nested inner 예측의 독립 잔차로 학습했다.',
      '', '### 단순 지표와의 추가 대조', '',
      '사후 반론으로 기존 예측값의 크기만 이용한 순위와 비교했다. 높은 값/낮은 값 두 방향을 모두 보존하며 새 모델 선택에 쓰지 않았다. EC는 높은 예측값 순위 AUC .858~.861로 새 위험분류 .831~.849보다 높았다. 온도는 낮은 예측값 순위 .629~.652로 새 분류 .570~.608보다 높았다. 따라서 이번 위험분류가 기존 예측값을 넘어선 구별 정보를 찾았다고 말할 수 없다. 단순 지표의 방향 비교도 사후 진단이고 보정식의 성능 개선은 아니다. risk_simple_comparator_v2.csv 및 별도 순위쌍/fsum/라이브러리 대조 PASS 참조.',
      '',
      '## 검산·반론','',
      f"독립 검산 {v['count']}체크 PASS. 핵심 모든 RMSE를 원행·math.fsum으로 재계산, prefix 평균·0시/현재/차이를 스칼라 계산으로 대조, 미래/다른온실 입력 변경 불변성, EC/온도 ±1buffer, 기존 ET 새학습 최대차 {v['baseline_ET_maxdiff']:.3g}를 확인했다. 위험 AUC도 모든 양성/음성 쌍의 순위 비교로 독립 검산했다.",
      '- 기각은 해당 사전 구성의 실패다. 하루 단위 모델·상태 분리·관계 학습 전체가 불가능하다는 증명은 아니다.',
      '- 공개 OOF와 같은 자료를 반복 사용했다. 새 독립 수집 자료의 일반화 검증이 아니며 동일 날씨·인접일 상관의 한계가 남는다.',
      '- 목표 변환의 비교는 LGB 직접정답 대조군을 포함했다. 온도 PFN을 gap 목표로 새 학습한 실험은 아니다.',
      '- rare 가중치는 훈련 정답으로만 계산했다. 검증 정답을 이용해 가중치·경로를 선택하지 않았다.',
      '- 온도 원가중치는 기존 TF 규칙을 그대로 재사용했다. 그 규칙의 입력 잡음 순위는 원 학습입력 전체에서 계산되므로 fold마다 새로 학습한 전처리라고 주장하지 않는다. 새 rare 가중치와 imputer/scaler는 fold 훈련만 사용했다.',
      '- ET 열 순서는 해당 실험 실행 전에 새 코드로 정정했고, 독립 검산에서 기존 ET 캐시가 이미 평활화된 출력임을 발견해 중복 평활화한 초기 수치를 폐기했다. 새 raw ET 예측을 한 번 평활화한 뒤 기존 캐시와 차이를 계산하는 최종 v3로 정정했다. 하이퍼파라미터/혼합비/학습은 변경하지 않았다. STATE20 체크포인트 중단/재개와 원본 로그를 보존했다.',
      '- 새 채택·제출 파일·평가 예측·EC 잠금 읽기/재채점 없음. 기존 실제 제출 W40G·EC 계절v2를 유지한다.','',
      '## 재현 및 저장','',
      'run.py는 최초 코드, run_v3.py는 열순서 정정 코드다. 같은 순서로 baseline → EC_DAY20 → EC_STATE20 → T_DIRECT20 → T_GAP20 → T_GAPSPLIT20 → EC_RARE_ET → T_RARE_GAP20 → EC_KNN20 → T_KNN20 → EC_H0CHANGE → risk를 실행한다. 그 뒤 analyze.py → verify.py → report.py. 기존 파일 보존을 위해 재실행 시 OUT을 새 버전 폴더로 바꾼다. 실행 환경은 env 부트스트랩·기존 저장 멤버/inner PFN 캐시를 요구한다.',
      '큰 OOF와 phase audit은 집/코덱스/local/priority_experiments_20261003_v1, 최종 ET 두 실험은 그 안 corrected_et_v3에 있다. 코드/프로토콜/로그/점수/검산/보고서는 이 analysis 폴더에 저장한다.']
    (HERE/'실험결과_보고서_v2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT_DONE')
if __name__=='__main__':main()
