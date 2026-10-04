"""Synthetic algebra only: Tweedie leaf optimum, curvature, prefix operator.

No competition input, saved predictions, score, model fit or model predict is read/run.
This is not the implementation of a new EC learner or a preregistered experiment.
"""
from pathlib import Path
import sys
import math
import json
import hashlib
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env  # first project import; only bootstrap, no model module


def mean(xs):
    return math.fsum(xs) / len(xs)


def prefix(xs):
    return [0.5 * x + 0.5 * math.fsum(xs[:i + 1]) / (i + 1) for i, x in enumerate(xs)]


def loss(y, margin):
    # rho=1.5: omit terms independent of the log margin.
    return 2 * y * math.exp(-0.5 * margin) + 2 * math.exp(0.5 * margin)


def leaf_derivative(eta, aa, bb, lam):
    return -aa * math.exp(-eta / 2) + bb * math.exp(eta / 2) + lam * eta


def bisect_monotone(aa, bb, lam):
    lo, hi = -1.0, 1.0
    while leaf_derivative(lo, aa, bb, lam) > 0:
        lo *= 2
    while leaf_derivative(hi, aa, bb, lam) < 0:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if leaf_derivative(mid, aa, bb, lam) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main():
    dest = HERE / '문헌수식_합성검산_v1.json'
    assert not dest.exists()
    checks, records = 0, []
    # A deliberately synthetic leaf; these numbers are not greenhouse observations.
    ys, mus = [.25, .25, 4.0], [.25, .25, .25]
    fs = [math.log(mu) for mu in mus]
    aa = math.fsum(y / math.sqrt(mu) for y, mu in zip(ys, mus))
    bb = math.fsum(math.sqrt(mu) for mu in mus)
    for y, f in zip(ys, fs):
        step = 1e-5
        g = math.exp(f / 2) - y * math.exp(-f / 2)
        h = .5 * (math.exp(f / 2) + y * math.exp(-f / 2))
        g_numeric = (loss(y, f + step) - loss(y, f - step)) / (2 * step)
        h_numeric = (loss(y, f + step) - 2 * loss(y, f) + loss(y, f - step)) / step**2
        assert abs(g - g_numeric) < 1e-8
        assert abs(h - h_numeric) < 5e-5
        checks += 2
    exact_unregularized = math.log(aa / bb)
    independent_unregularized = math.log(mean(ys) / mus[0])
    assert abs(exact_unregularized - independent_unregularized) < 1e-12
    assert abs(leaf_derivative(exact_unregularized, aa, bb, 0)) < 1e-12
    checks += 2
    for lam in (0.0, 1.0):
        newton = (aa - bb) / (.5 * (aa + bb) + lam)
        exact = bisect_monotone(aa, bb, lam)
        assert abs(leaf_derivative(exact, aa, bb, lam)) < 1e-12
        if lam == 0:
            assert abs(exact - exact_unregularized) < 1e-12
            assert abs(newton - 2 * (aa / bb - 1) / (aa / bb + 1)) < 1e-12
        checks += 1 if lam else 3
        records.append(dict(lambda_l2=lam, newton_log_increment=newton, exact_log_increment=exact,
                            derivative_at_newton=leaf_derivative(newton, aa, bb, lam),
                            derivative_at_exact=leaf_derivative(exact, aa, bb, lam)))
    # The conditional expected Tweedie loss is minimized at mu=E[Y], not at the median.
    expected_mean = mean(ys)
    g_at_mean = math.fsum(math.sqrt(expected_mean) - y / math.sqrt(expected_mean) for y in ys)
    assert abs(g_at_mean) < 1e-12
    checks += 1
    # Prefix smoothing preserves any constant level, including a constant level error.
    levels = [1.0] * 24
    assert prefix(levels) == levels
    shifted = [x - .3 for x in levels]
    assert max(abs(a - b) for a, b in zip(prefix(shifted), shifted)) < 1e-12
    checks += 2
    # Verify exact coupled gradient using an independent finite difference.
    n = 4
    ss = [[(.5 if i == j else 0) + (.5 / (i + 1) if j <= i else 0) for j in range(n)] for i in range(n)]
    qq, yy = [.2, .4, .1, .6], [.3, .5, .2, .5]
    sq = [math.fsum(s * x for s, x in zip(row, qq)) for row in ss]
    residual = [a - b for a, b in zip(sq, yy)]
    gradient = [math.fsum(ss[i][j] * residual[i] for i in range(n)) for j in range(n)]
    hessian = [[math.fsum(ss[i][j] * ss[i][k] for i in range(n)) for k in range(n)] for j in range(n)]
    def coupled_loss(q):
        p = [math.fsum(s * x for s, x in zip(row, q)) for row in ss]
        return .5 * math.fsum((a - b)**2 for a, b in zip(p, yy))
    for j in range(n):
        step = 1e-5
        plus, minus = qq.copy(), qq.copy()
        plus[j] += step
        minus[j] -= step
        numeric = (coupled_loss(plus) - coupled_loss(minus)) / (2 * step)
        assert abs(numeric - gradient[j]) < 1e-10
        checks += 1
    assert hessian[0][1] != 0
    checks += 1
    catalog = (ROOT / '공용/데이터_단서_카탈로그.md').read_text(encoding='utf-8')
    last = re.findall(r'^\| (6\.\d+) \|', catalog, flags=re.M)[-1]
    result = dict(status='PASS', scope='synthetic algebra only; zero EC data or score', checks=checks,
        catalog_latest_at_execution=last, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        synthetic_leaf=dict(y=ys, mu=mus, aa=aa, bb=bb, expected_mean=expected_mean,
                            exact_unregularized_log_increment=exact_unregularized, comparisons=records),
        prefix_constant_level_preserved=True, coupled_gradient=gradient,
        coupled_hessian=hessian, coupled_hessian_is_diagonal=False,
        experiment_fit=0, experiment_predict=0, EC_score=0,
        limits=['No installed LightGBM source or actual fitted leaf was audited.',
                'Exact scalar leaf minimization does not establish EC RMSE improvement.',
                'Gradient/Hessian proof omits final clipping; it is not a full clipped-loss optimizer.'])
    with dest.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
