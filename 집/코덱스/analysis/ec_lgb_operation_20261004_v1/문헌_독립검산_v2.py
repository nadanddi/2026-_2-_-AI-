"""Final literature checks: synthetic algebra and own artifact hashes only.

Reads no competition input, model checkpoint, score CSV, train_y, or actual prediction.
Preserves previous files; this run creates fresh v2 output files exclusively.
"""
from pathlib import Path
import sys
import math
import json
import hashlib
import ast
import re
from decimal import Decimal, localcontext
from fractions import Fraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env  # first project import; bootstrap only


def main():
    output = HERE / '문헌_독립검산_v2.json'
    scope_output = HERE / '문헌_읽기범위_v1.json'
    report_v2 = HERE / '문헌후속_v2.md'
    for path in (output, scope_output, report_v2):
        assert not path.exists(), path
    original = json.loads((HERE / '문헌수식_합성검산_v1.json').read_text(encoding='utf-8'))
    assert original['status'] == 'PASS' and original['checks'] == 20
    assert original['experiment_fit'] == original['experiment_predict'] == original['EC_score'] == 0
    leaf = original['synthetic_leaf']
    assert leaf['y'] == [.25, .25, 4.] and leaf['mu'] == [.25, .25, .25]
    records = {row['lambda_l2']: row for row in leaf['comparisons']}
    checks = 4

    # Decimal is independent of the original binary fsum and bisection path.
    with localcontext() as context:
        context.prec = 50
        ds = [Decimal('0.25'), Decimal('0.25'), Decimal('4')]
        aa = sum(y / Decimal('.25').sqrt() for y in ds)
        bb = sum(Decimal('.25').sqrt() for _ in ds)
        assert aa == 9 and bb == Decimal('1.5')
        exact0 = (aa / bb).ln()
        independent0 = ((sum(ds) / len(ds)) / Decimal('.25')).ln()
        assert abs(exact0 - independent0) < Decimal('1e-45')
        assert abs(float(exact0) - records[0.0]['exact_log_increment']) < 1e-12
        checks += 3
        decimal_newton = {}
        for lam in (Decimal(0), Decimal(1)):
            value = (aa - bb) / ((aa + bb) / 2 + lam)
            assert abs(float(value) - records[float(lam)]['newton_log_increment']) < 1e-12
            decimal_newton[str(lam)] = str(value)
            checks += 1

    # Newton-Raphson solves the original monotone root independently of bisection.
    t = 1.0
    for _ in range(20):
        derivative = -9 * math.exp(-t / 2) + 1.5 * math.exp(t / 2) + t
        curvature = 4.5 * math.exp(-t / 2) + .75 * math.exp(t / 2) + 1
        t -= derivative / curvature
    assert abs(t - records[1.0]['exact_log_increment']) < 1e-12
    assert abs(-9 * math.exp(-t / 2) + 1.5 * math.exp(t / 2) + t) < 1e-12
    checks += 2

    # Exact rational prefix matrices compare against direct running means.
    def smoothing_matrix(n):
        return [[(Fraction(1, 2) if h == k else Fraction(0))
                 + (Fraction(1, 2 * (h + 1)) if k <= h else Fraction(0))
                 for k in range(n)] for h in range(n)]
    s24 = smoothing_matrix(24)
    for level in (Fraction(1), Fraction(7, 10), Fraction(-3, 10)):
        vals = [level] * 24
        by_matrix = [sum(a * b for a, b in zip(row, vals)) for row in s24]
        by_running = []
        running = Fraction(0)
        for h, value in enumerate(vals):
            running += value
            by_running.append(value / 2 + running / (2 * (h + 1)))
        assert by_matrix == by_running == vals
        checks += 1
    s4 = smoothing_matrix(4)
    h01 = sum(row[0] * row[1] for row in s4)
    assert h01 == Fraction(133, 576)
    assert abs(float(h01) - original['coupled_hessian'][0][1]) < 1e-15
    checks += 2

    # Final component coordinate: H_q=.24^2 S^T S, distinct from unscaled unit operator.
    mix = Fraction(6, 25)
    h01_scaled = mix * mix * h01
    assert h01_scaled == Fraction(133, 10000)
    checks += 1

    # Independently check exp chain rule with final coupled loss, using synthetic inputs.
    fs = [-1.2, -.6, -.9, -.3]
    b = [.2, .1, .3, .2]
    y = [.3, .5, .2, .5]
    q = [math.exp(f) for f in fs]
    p = [bi + .24 * qi for bi, qi in zip(b, q)]
    z = [math.fsum(float(si) * pi for si, pi in zip(row, p)) for row in s4]
    r = [zi - yi for zi, yi in zip(z, y)]
    gq = [.24 * math.fsum(float(s4[i][j]) * r[i] for i in range(4)) for j in range(4)]
    gf = [qi * gi for qi, gi in zip(q, gq)]
    def final_loss(margins):
        values = [bi + .24 * math.exp(fi) for bi, fi in zip(b, margins)]
        predictions = [math.fsum(float(si) * pi for si, pi in zip(row, values)) for row in s4]
        return .5 * math.fsum((pi - yi)**2 for pi, yi in zip(predictions, y))
    for j in range(4):
        step = 1e-5
        plus, minus = fs.copy(), fs.copy()
        plus[j] += step
        minus[j] -= step
        numeric = (final_loss(plus) - final_loss(minus)) / (2 * step)
        assert abs(numeric - gf[j]) < 1e-10
        checks += 1

    # Correct the unit-operator versus .24-scaled Hessian wording in a NEW version.
    report = (HERE / '문헌후속_v1.md').read_text(encoding='utf-8')
    assert report.count('합성 4시간 유한차분 및 비대각 H[0,1]=.2309027778 대조.') == 1
    report = report.replace('# EC 잎값·손실·평활화 문헌 후속 v1', '# EC 잎값·손실·평활화 문헌 후속 v2', 1)
    report = report.replace(
        '합성 4시간 유한차분 및 비대각 H[0,1]=.2309027778 대조.',
        '합성 4시간 유한차분 및 단위 prefix 연산 SᵀS의 비대각 [0,1]=.2309027778 대조. 실제 .24 혼합 계수를 곱한 q 좌표 Hessian의 해당 값은 .0133이며, F 좌표에서는 추가 chain rule이 필요.')
    report += ('\n\nv2 독립 재검산: `문헌_독립검산_v2.py`는 Decimal 합계·Newton-Raphson 근·'
               'Fraction prefix 행렬·exp chain rule 유한차분을 별도로 실행했다. v1의 표에서 '
               '단위 SᵀS와 실제 .24² SᵀS의 수치 범위를 명확히 구분했다. '
               '기존 v1 및 합성 결과는 수정하지 않았다. 실제 EC 자료·점수 읽기/재계산 0이며 '
               '코드·결과·읽기 범위는 `문헌_독립검산_v2.json`, `문헌_읽기범위_v1.json`에 보존했다.\n')
    report_v2.write_text(report, encoding='utf-8')
    checks += 1

    hashes = {}
    for filename in ('검산_문헌수식_v1a.py', '문헌수식_합성검산_v1.json',
                     '문헌_독립검산_v2.py', '문헌후속_v1.md', '문헌후속_v2.md'):
        path = HERE / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'))
            checks += 1
    assert hashes['검산_문헌수식_v1a.py'] == original['source_sha256']
    checks += 1
    catalog = (ROOT / '공용/데이터_단서_카탈로그.md').read_text(encoding='utf-8')
    latest = re.findall(r'^\| (6\.\d+) \|', catalog, flags=re.M)[-1]
    scope = dict(
        scope='primary literature and official tag source only; actual EC data/score zero',
        papers=[
            dict(title='TDboost', url='https://arxiv.org/pdf/1508.06378v3',
                 version='v3; arXiv 2016-04-20', read='4.1 equations 11-19; 5; 6.2-6.3',
                 limits='not full appendix or author package execution; no EC RMSE evidence'),
            dict(title='Gradient and Newton Boosting', url='https://arxiv.org/pdf/1808.03064v7',
                 version='v7; arXiv 2020-10-20', read='2.1.1-2.1.5; 2.4-2.4.1; 3; 4.2-4.3; 5',
                 limits='not full appendix recomputation; regression metric mostly NLL'),
            dict(title='Autocalibration and Tweedie-dominance', url='https://arxiv.org/pdf/2103.03635v2',
                 version='v2; arXiv 2021-07-09', read='3-5.2; 6; 7',
                 limits='no author code execution; local calibration overlaps existing attempts')],
        official_sources=[
            'https://github.com/lightgbm-org/LightGBM/blob/v4.7.0/src/objective/regression_objective.hpp',
            'https://github.com/lightgbm-org/LightGBM/blob/v4.7.0/include/LightGBM/objective_function.h',
            'https://github.com/lightgbm-org/LightGBM/blob/v4.7.0/src/treelearner/feature_histogram.hpp'],
        duplicate_catalog=['6.143', '6.177', '6.257', '6.259-6.266', '6.267-6.268', '6.271', '6.292'],
        catalog_latest_at_final_guard=latest,
        local_dll_build_audit=False, actual_data_reads=0, fit=0, predict=0, ec_score=0,
        report='문헌후속_v2.md', artifact_hashes=hashes)
    with scope_output.open('x', encoding='utf-8') as handle:
        json.dump(scope, handle, ensure_ascii=False, indent=2)
    result = dict(status='PASS_SYNTHETIC_ONLY', checks=checks,
        decimal_newton=decimal_newton, newton_raphson_lambda1=t,
        prefix_h01_unit=str(h01), prefix_h01_unit_float=float(h01),
        prefix_h01_scaled_q=str(h01_scaled), prefix_h01_scaled_q_float=float(h01_scaled),
        exp_chain_rule_gradient=gf, sha256=hashes,
        catalog_latest_at_final_guard=latest, actual_data_reads=0, fit=0, predict=0, ec_score=0,
        limits=['No actual fitted EC leaf or DLL build audited.',
                'Final loss derivatives are pre-clip synthetic results only.',
                'No performance or causal claim.'])
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=result['status'], checks=checks,
                          latest=latest, output=str(output), report=str(report_v2)), ensure_ascii=False))


if __name__ == '__main__':
    main()
