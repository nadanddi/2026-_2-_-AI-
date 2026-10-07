import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))
result = read('BLK_diagnostic_results_v1.json')
check = read('BLK_score_independent_crosscheck_v1.json')
assert check['status'] == 'PASS'
assert check['result_sha256'] == hashlib.sha256((HERE/'BLK_diagnostic_results_v1.json').read_bytes()).hexdigest()
lines = ['# BLK 앞뒤 정답 연결 첫 진단 — 전수 재점검 중간 결과', '',
         '6.374에서 내부 query 통계 fit 위험을 분리한 학습 전용 cached CPU 기준선에 대한 진단이다. 과거 GPU OOF·제출14와 출력 동일성을 입증한 실험이 아니다. 공개 EC 1440행, 60기록일, 8블록, 시드47/1414/6464를 사용했다. 블록과 방법은 정답을 보기 전에 고정했다.', '',
         '## 전체 비교', '',
         '표의 RMSE는 세 시드별 RMSE의 산술평균이다. 차이는 후보−기준선이며 양수는 악화다. 두 scope×세 방법, 총6variant에 alpha=.025/6을 적용했다.', '',
         '|범위|방법|기준 RMSE|후보 RMSE|차이|P(worse)|BLK 선별|',
         '|---|---|---:|---:|---:|---:|---|']
for entry in result['results']:
    cells = [c for c in entry['cells'] if c['segment']=='전체']
    base = sum(c['baseline_RMSE'] for c in cells)/3
    candidate = sum(c['candidate_RMSE'] for c in cells)/3
    lines.append(f"|{entry['scope']}|{entry['method']}|{base:.8f}|{candidate:.8f}|{candidate-base:+.8f}|{entry['p_worse']:.6f}|{'PASS' if entry['BLK_screen_pass'] else 'FAIL'}|")
lines += ['', '## 주 비교 BLK_QUERY_ROLE의 구간 차이', '',
          '|방법|구간|행수|기준 RMSE|후보 RMSE|차이|', '|---|---|---:|---:|---:|---:|']
for entry in result['results']:
    if entry['scope'] != 'BLK_QUERY_ROLE' or entry['method']=='CHAIN_PREFIX_GUARD':
        continue
    for segment in ['일반','고EC','앞','가운데','뒤']:
        cells=[c for c in entry['cells'] if c['segment']==segment]
        base=sum(c['baseline_RMSE'] for c in cells)/3
        candidate=sum(c['candidate_RMSE'] for c in cells)/3
        lines.append(f"|{entry['method']}|{segment}|{cells[0]['rows']}|{base:.8f}|{candidate:.8f}|{candidate-base:+.8f}|")
lines += ['', '## 확인한 주장과 반론', '',
          '- 단순 양끝 보간·과거 끝 혼합은 이 BLK에서 채택 근거가 없다(신뢰도 높음: 이 등록 실험의 선별 판정). 두 방법 모두 전체 RMSE가 세 시드에서 악화하고 선별 P 기준을 통과하지 않는다. 반론: 다른 연결 또는 보정 수식은 미시험이므로 끝 정답의 정보가 무용하다는 결론은 아니다.',
          '- 양끝 보간은 일반 행에서 악화·고EC 행에서 개선하는 상반된 결과다(신뢰도 중간: 이 표본의 묘사). 고EC는 3일/72행뿐이다. 사후 고EC 구분을 예측 시각에서 아는 입력으로 쓰거나 계수를 고르는 근거로 삼지 않는다.',
          '- 좁은 mutual-component CHAIN_PREFIX_GUARD는 active0/1440으로 세 시드×두 범위에서 기준선과 동일하다(신뢰도 높음: 저장값 독립 대조). 반론: 이는 원 CH2의 slope 보정 Hungarian 학습 참조 사슬을 시험한 결과가 아니다. CH2 전체 단서 기각 금지.',
          '- 양끝 보간의 가운데 구간 개선·앞/뒤 악화는 사전 고정 위치 정의의 기술통계다(신뢰도 중간). 위치만으로 보정 적용 구간을 사후 고르면 새 후보이며 다중비교에 추가해야 한다.', '',
          '## 검증 근거와 한계', '',
          '- 실제 registered receipt/assemble/verify/score 단계 각각 exit0 및 source 불변. 전체 raw mix→shrink1회→clip 독립 검산, 최종 causal288검사. cached PFN 네 문맥 각18개 수치 감사와 내부 fit/cache 불변 확인.',
          '- 독립 Decimal60 직접 ID손실로 모든 셀 RMSE, 블록 손실 합계, explicit randrange 농장별4블록×20k 표집의 P와 CI를 재검산했다. 실제 gate 독립 비평도 실시했다. 모델의 독립 재학습을 뜻하지 않는다.',
          '- 8블록·같은 참조 학습자료·반복 사용한 공개 정답의 한계가 있다. 개별95% CI는 기술적이며 본페로니 조정 CI가 아니다. 새 layout/seed도 새 정답을 제공하지 않는다.',
          '- 선택 BLK는 모두 1차 기록이다. RAW_PASS는 SG2가 skip되고 QUERY_ROLE은 별도 평가 역할 적용이다. 실제 2차 일반화는 미입증이다.',
          '- 이번 결과로 BLK를 기존 TM/P2LOO/EL1 대신 주 검증기로 채택하지 않는다. 기존 검증기 전체 효과 시험, 도메인24→문헌24→데이터142, 최초 미사용 seed/layout 확정1회, 최종 정렬 보고서는 아직 미완료다.', '',
          '## 근거 파일', '',
          '- BLK_diagnostic_results_v1.json / BLK_diagnostic_cells_v1.csv',
          '- BLK_score_independent_crosscheck_v1.json / crosscheck_BLK_score_v1.py',
          '- BLK_verified_baseline_receipt_v4.json / critique_BLK_actual_gate_v1.md',
          '- BLK_method_registration_v4.json / BLK_scorer_spec_v1.json',
          '- PROGRESS.md: 재개 지점. 다음 bounded 사슬 후속 여부는 독립 사후 비평으로 결정하되, 방법은 새 실행 전에 봉인한다.']
out = HERE/'BLK_첫진단_보고서_v1.md'
assert not out.exists()
out.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Verified BLK diagnostic report saved')
