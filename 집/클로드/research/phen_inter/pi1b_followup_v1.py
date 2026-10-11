# -*- coding: utf-8 -*-
"""PI1b 후속: (a) 기록별 배지 결합 계수와 EC/잔차 관계의 계절·EC수준 통제, (b) 2차 사건 반응을 1·2차 통합 대조 풀로 재계산(같은 농장)."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
OUT = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(1)
FE = pd.read_csv(os.path.join(OUT, 'pi1_record_response_v1.csv'))
FE['day'] = FE.rec.str[4:7].astype(int)
def resid_on(df, y, xs):
    d = df[[y] + xs].dropna(); Z = np.column_stack([np.ones(len(d))] + [d[x] for x in xs])
    return pd.Series(d[y] - Z @ np.linalg.lstsq(Z, d[y], rcond=None)[0], index=d.index)
def seas(df, col, win=15):
    # 같은 농장 안 day 기준 ±win일 이동중앙값 제거(계절 성분)
    out = pd.Series(np.nan, index=df.index)
    for fm, g in df.groupby('farm'):
        for i, r in g.iterrows():
            nb = g[(abs(g.day - r.day) <= win) & (g.index != i)][col]
            out[i] = r[col] - nb.median() if nb.notna().sum() >= 3 else np.nan
    return out
for c in ['k_lag', 'k_lvl', 'ec25_mean', 'absres', 'res_mean', 'res_0', 'night_gap']:
    FE[c + '_s'] = seas(FE, c)
def perm(a, b, grp, n=2000):
    ok = np.isfinite(a) & np.isfinite(b); a, b, grp = a[ok], b[ok], grp[ok]
    obs = spearmanr(a, b)[0]; c = 0
    for _ in range(n):
        bb = b.copy()
        for gv in np.unique(grp):
            ix = np.where(grp == gv)[0]; bb[ix] = rng.permutation(bb[ix])
        c += abs(spearmanr(a, bb)[0]) >= abs(obs)
    return round(obs, 3), round((c + 1) / (n + 1), 4), int(ok.sum())
rows = []
for k in ['k_lag', 'k_lvl', 'night_gap']:
    for t in ['ec25_mean', 'absres', 'res_mean', 'res_0']:
        for ph in (1, 2):
            g = FE[FE.ph == ph]
            raw = perm(g[k].values, g[t].values, g.farm.values)
            sea = perm(g[k + '_s'].values, g[t + '_s'].values, g.farm.values)
            rows.append(dict(resp=k, target=t, ph=ph, raw_rho=raw[0], raw_p=raw[1], seas_rho=sea[0], seas_p=sea[1], n=sea[2]))
    # absres controlling ec level (rank partial)
    for ph in (1, 2):
        g = FE[FE.ph == ph].dropna(subset=[k, 'absres', 'ec25_mean'])
        rk = lambda s: rankdata(s)
        Z = np.column_stack([np.ones(len(g)), rk(g.ec25_mean), (g.farm == 'F13').astype(float)])
        ra = rk(g[k]) - Z @ np.linalg.lstsq(Z, rk(g[k]), rcond=None)[0]
        rb = rk(g.absres) - Z @ np.linalg.lstsq(Z, rk(g.absres), rcond=None)[0]
        rows.append(dict(resp=k, target='absres|ec25수준통제', ph=ph, raw_rho=round(np.corrcoef(ra, rb)[0, 1], 3), n=len(g)))
R = pd.DataFrame(rows); print(R.to_string(index=False))
# high-EC vs low-EC coupling contrast
for ph in (1, 2):
    g = FE[FE.ph == ph]
    q = g.ec25_mean.quantile([.33, .67]).values
    print(ph, 'k_lag 저EC', g[g.ec25_mean <= q[0]].k_lag.median().round(3), '중', g[(g.ec25_mean > q[0]) & (g.ec25_mean <= q[1])].k_lag.median().round(3), '고', g[g.ec25_mean > q[1]].k_lag.median().round(3),
          '| k_lvl', g[g.ec25_mean <= q[0]].k_lvl.median().round(3), g[g.ec25_mean > q[1]].k_lvl.median().round(3))
# causal version: k_lag estimated from hours 0..h only (h=8, 12) vs whole-day; does it still relate to EC?
X = pd.read_csv(os.path.join(env.DATA, 'train_X.csv')); Y = pd.read_csv(os.path.join(env.DATA, 'train_y.csv'))
X = X[X.row_id.str[:3].isin(['F13', 'F47'])].merge(Y, on='row_id'); X['rec'] = X.row_id.str[:7]; X['hour'] = X.row_id.str[-2:].astype(int)
X = X.sort_values(['rec', 'hour'])
for H in (8, 12, 23):
    ks = {}
    for rec, g in X.groupby('rec'):
        g = g[g.hour <= H]
        din = g.in_temp.diff().values; dsub = g.sub_temp.diff().values
        Zl = np.column_stack([pd.Series(din).shift(k).values for k in (1, 2, 3)])
        m = np.isfinite(dsub) & np.isfinite(Zl).all(1)
        if m.sum() >= 5:
            Z = np.column_stack([np.ones(m.sum()), Zl[m]]); b = np.linalg.lstsq(Z, dsub[m], rcond=None)[0]; ks[rec] = b[1:].sum()
    FE['kc'] = FE.rec.map(ks)
    for ph in (1, 2):
        g = FE[FE.ph == ph]
        print(f'0..{H}시만 쓴 k_lag', ph, 'vs ec25_mean', perm(g.kc.values, g.ec25_mean.values, g.farm.values, 1000), 'vs ec25_0(0시 EC)', perm(g.kc.values, g.ec25_0.values, g.farm.values, 1000), 'vs res_0', perm(g.kc.values, g.res_0.values, g.farm.values, 1000), 'vs absres', perm(g.kc.values, g.absres.values, g.farm.values, 1000))
