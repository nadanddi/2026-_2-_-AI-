# -*- coding: utf-8 -*-
"""Follow-up of hb1: the only set with any positive sign was C2_midcore (gap + intermediate-day
daily-mean drift of Tin, rad, heat, Hin, transp vs anchor day) for TEMP.  Check:
 (1) fit on pass 1 only (all k>=2 pooled, + k), judge on pass 2 (no CV leakage across passes);
 (2) single-feature increments; (3) cluster bootstrap by chain component of the pass-2 RMSE diff;
 (4) restricted to same-ID paths.  Reads hb1_pairs_*.csv + rebuilds hourly residuals."""
import os, sys
sys.path.insert(0, r'C:\work\farmai\집\클로드\research')
import env, importlib.util
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
RES = r'C:\work\farmai\집\클로드\research'; OUT = os.path.join(RES, 'agentX_bridge')
rng = np.random.default_rng(0)
CORE = ['mid_Tin', 'mid_rad', 'mid_heat', 'mid_Hin', 'mid_transp']
for target in ['temp', 'ec']:
    P = pd.read_csv(os.path.join(OUT, f'hb1_pairs_{target}.csv'))
    P = P[P.k >= 2].copy(); P['kk'] = P.k
    tr = P[P.pass2 == 0]; te = P[P.pass2 == 1]
    print(f'\n===== {target}: fit pass1 n={len(tr)}, judge pass2 n={len(te)} (targets {te.t.nunique()})')

    def fitpred(fs, ycol, a=tr, b=te):
        sc = StandardScaler().fit(a[fs].values); m = Ridge(alpha=1.0).fit(sc.transform(a[fs].values), a[ycol].values)
        return m.predict(sc.transform(b[fs].values)), dict(zip(fs, np.round(m.coef_, 4)))

    for ycol in ['rm', 'r0']:
        yv = te[ycol].values
        pg, cg = fitpred(['gap', 'kk'], ycol)
        res = {'none': np.mean(yv ** 2), 'gap': np.mean((yv - pg) ** 2)}
        pc, cc = fitpred(['gap', 'kk'] + CORE, ycol); res['core'] = np.mean((yv - pc) ** 2)
        for f in CORE:
            p1, _ = fitpred(['gap', 'kk', f], ycol); res['+' + f] = np.mean((yv - p1) ** 2)
        print(f'  {ycol}: pass2 MSE of day-level residual ->', {k: round(v, 4) for k, v in res.items()})
        print(f'     core coefs (std units):', cc)
        # cluster bootstrap of core vs gap SSE diff by chain component
        d = (yv - pc) ** 2 - (yv - pg) ** 2; g = te.comp.values; ug = np.unique(g)
        S = np.array([d[g == u].sum() for u in ug]); B = []
        for _ in range(2000):
            B.append(S[rng.integers(0, len(ug), len(ug))].sum())
        B = np.array(B)
        print(f'     core-gap SSE diff {d.sum():+.3f}  share worse in bootstrap P(diff>=0)={np.mean(B >= 0):.3f}  comps={len(ug)}  per-target improved {np.mean(d < 0):.2f}')
        # same-ID only
        m = te.sameid.values.astype(bool)
        print(f'     same-ID paths only (n={m.sum()}): gap MSE {np.mean((yv[m]-pg[m])**2):.4f}  core MSE {np.mean((yv[m]-pc[m])**2):.4f}  none {np.mean(yv[m]**2):.4f}')
        # by k
        for k in range(2, 6):
            mk = te.k.values == k
            print(f'     k={k} n={mk.sum()}: none {np.mean(yv[mk]**2):.4f} gap {np.mean((yv[mk]-pg[mk])**2):.4f} core {np.mean((yv[mk]-pc[mk])**2):.4f}')
    # reverse direction: fit pass2, judge pass1 (stability of sign)
    for ycol in ['rm', 'r0']:
        yv = tr[ycol].values
        pg, _ = fitpred(['gap', 'kk'], ycol, te, tr); pc, _ = fitpred(['gap', 'kk'] + CORE, ycol, te, tr)
        print(f'  reverse (fit p2 -> judge p1) {ycol}: none {np.mean(yv**2):.4f} gap {np.mean((yv-pg)**2):.4f} core {np.mean((yv-pc)**2):.4f}')
