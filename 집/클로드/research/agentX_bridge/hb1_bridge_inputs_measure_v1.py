# -*- coding: utf-8 -*-
"""H-BRIDGE (agentX): when the same-house labelled anchor is k days back, can the
INPUTS of the anchor day / intermediate days / target 0h explain the drift of the
target's day-level residual beyond the anchor gap?  Training records only, truth links
(linker_v3 truth.pkl a1), DIAG10 OOF (temp: mean of 4 cells; EC: CUR sg mean of seeds).
CV grouped by chain component (5 folds), ridge on standardized features (alpha fixed 1.0,
not tuned).  Descriptive / oracle measurement only - no model, no eval data used."""
import os, sys
sys.path.insert(0, r'C:\work\farmai\집\클로드\research')
import env, importlib.util, pickle
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
np.random.seed(0)
RES = r'C:\work\farmai\집\클로드\research'
sys.argv = ['x']
s = importlib.util.spec_from_file_location('ta2', os.path.join(RES, 'ta2_anchor_debiased_v1.py')); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
M, cells = A2.A1.frames()['DIAG10']
M = M[['row_id', 'farm', 'day', 'hour', 'sub_temp']].copy(); M['q'] = np.mean(np.vstack(cells), 0)
E = pd.read_csv(r'C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv')
E = E[(E.tag == 'CUR') & (E.validator == 'DIAG10')].groupby('row_id').agg(q=('sg', 'mean'), y=('sub_ec', 'first')).reset_index()

T = pd.read_pickle(os.path.join(RES, 'local/linker_v3/truth.pkl'))
R = pickle.load(open(os.path.join(RES, 'local/linker_v3/base.pkl'), 'rb'))
r = R['recs'].reset_index(drop=True)
ids = [f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)]
N = len(r)
X = pd.read_csv(os.path.join(env.DATA, 'train_X.csv'))
X = X[X.row_id.str[:3].isin(['F13', 'F47'])].copy(); X['rid'] = X.row_id.str[:7]; X['h'] = X.row_id.str[-2:].astype(int)
VARS = ['out_temp', 'out_hum', 'out_rad', 'out_wspd', 'in_temp', 'in_hum', 'in_co2', 'act_vent', 'act_shade', 'act_thermal', 'act_heating', 'act_circfan', 'act_co2', 'act_fog']
IDX = {k: i for i, k in enumerate(ids)}


def tens(df, col, key):
    a = np.full((N, 24), np.nan)
    for rid, g in df.groupby(key):
        if rid in IDX:
            a[IDX[rid], g.h.values] = g[col].values
    return a


inp = {v: tens(X, v, 'rid') for v in VARS}
M['rid'] = M.row_id.str[:7]; M['h'] = M.hour
yT = tens(M, 'sub_temp', 'rid'); qT = tens(M, 'q', 'rid')
E['rid'] = E.row_id.str[:7]; E['h'] = E.row_id.str[-2:].astype(int)
yE = tens(E, 'y', 'rid'); qE = tens(E, 'q', 'rid')
istrain = (r.split == 'train').values
print('records', N, 'train', istrain.sum(), 'tempOOF days', np.isfinite(qT).all(1).sum(), 'ecOOF days', np.isfinite(qE).all(1).sum())


def dm(v):
    return np.nanmean(inp[v], 1)


D = pd.DataFrame({'Tin': dm('in_temp'), 'Tout': dm('out_temp'), 'rad': np.nansum(inp['out_rad'], 1) / 1000.0,
                  'Hin': dm('in_hum'), 'Hout': dm('out_hum'), 'heat': dm('act_heating'), 'vent': dm('act_vent'),
                  'therm': dm('act_thermal'), 'fog': dm('act_fog'), 'shade': dm('act_shade'), 'co2a': dm('act_co2'),
                  'fan': dm('act_circfan'), 'co2': dm('in_co2')})
es = 0.6108 * np.exp(17.27 * inp['in_temp'] / (inp['in_temp'] + 237.3))
D['vpd'] = np.nanmean(es * (1 - inp['in_hum'] / 100.0), 1)
D['transp'] = np.nansum(inp['out_rad'] * (1 - inp['in_hum'] / 100.0), 1) / 1000.0
DCOL = list(D.columns)
D = D.fillna(0.0)

a1 = np.full(N, -1); a1[T.b.values] = T.a1.values
cert = np.zeros(N, bool); cert[T.b.values] = T.cert.values
par = list(range(N))


def fnd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x


for b, a in zip(T.b.values, T.a1.values):
    par[fnd(b)] = fnd(a)
comp = np.array([fnd(i) for i in range(N)])
farm = r.farm.values; day = r.day.values


def build(target, KMAX=5):
    y, q = (yT, qT) if target == 'temp' else (yE, qE)
    ok = np.isfinite(y).all(1) & np.isfinite(q).all(1) & istrain
    rows = []
    for t in range(N):
        if not ok[t]:
            continue
        path = [t]; allcert = True; sameid = True
        for k in range(1, KMAX + 1):
            p = path[-1]; a = a1[p]
            if a < 0 or a in path or not istrain[a]:
                break
            allcert &= bool(cert[p]); sameid &= bool(farm[a] == farm[t])
            path.append(a)
            if not ok[a]:
                continue
            mids = path[1:-1]
            gap = y[a, 23] - q[t, 0]
            ra = y[a] - q[a]
            f = dict(t=t, a=a, k=k, comp=comp[t], pass2=int(day[t] >= 179), cert=allcert, sameid=sameid,
                     r0=y[t, 0] - q[t, 0], rm=np.mean(y[t] - q[t]), gap=gap,
                     a_trend=ra[23] - ra[0], a_rm=ra.mean(), a_r23=ra[23], a_eve=ra[18:24].mean() - ra[:6].mean(),
                     dT0=inp['in_temp'][t, 0] - inp['in_temp'][a, 23], dH0=inp['in_hum'][t, 0] - inp['in_hum'][a, 23],
                     dTo0=inp['out_temp'][t, 0] - inp['out_temp'][a, 23])
            for c in DCOL:
                mv = D[c].values[mids].mean() if mids else D[c].values[a]
                f['mid_' + c] = mv - D[c].values[a]
                f['cum_' + c] = D[c].values[mids + [a]].sum()
                f['tfull_' + c] = D[c].values[t] - D[c].values[a]
            f['dyT'] = np.mean(y[t]); f['dya'] = np.mean(y[a])
            rows.append(f)
    return pd.DataFrame(rows)


def cvr2(df, feats, ycol, nf=5):
    Xm = np.nan_to_num(df[feats].values.astype(float)); yv = df[ycol].values; pred = np.zeros(len(df))
    g = df.comp.values; nf = min(nf, len(np.unique(g)))
    for tr, te in GroupKFold(nf).split(Xm, yv, g):
        sc = StandardScaler().fit(Xm[tr]); m = Ridge(alpha=1.0).fit(sc.transform(Xm[tr]), yv[tr])
        pred[te] = m.predict(sc.transform(Xm[te]))
    return 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2), pred


GAP = ['gap']
SETS = {
    'A_anchorown': GAP + ['a_trend', 'a_rm', 'a_r23', 'a_eve'],
    'B_0h': GAP + ['dT0', 'dH0', 'dTo0'],
    'C_mid': GAP + ['mid_' + c for c in DCOL],
    'C2_midcore': GAP + ['mid_Tin', 'mid_rad', 'mid_heat', 'mid_Hin', 'mid_transp'],
    'D_cum': GAP + ['cum_' + c for c in ['rad', 'transp', 'Tin', 'heat', 'vent', 'fog']],
    'E_allok': GAP + ['a_trend', 'a_rm', 'a_r23', 'a_eve', 'dT0', 'dH0', 'dTo0'] + ['mid_' + c for c in DCOL],
    'X_tfull': GAP + ['tfull_' + c for c in DCOL],
}
out = []; oracle = []
for target in ['temp', 'ec']:
    P = build(target)
    print(f'\n===== {target}: pairs {len(P)}  by k', P.groupby('k').size().to_dict(), ' pass2 by k', P.groupby('k').pass2.sum().to_dict())
    yy = (yT if target == 'temp' else yE); qq = (qT if target == 'temp' else qE)
    for k in range(1, 6):
        for ps in [0, 1, 'all']:
            for ycol in ['rm', 'r0']:
                sub = P[(P.k == k) & ((P.pass2 == ps) if ps != 'all' else True)]
                if len(sub) < 15:
                    continue
                base, pb = cvr2(sub, GAP, ycol)
                row = dict(target=target, k=k, pass_=ps, y=ycol, n=len(sub), sameid=sub.sameid.mean(), cert=sub.cert.mean(), r2_gap=base)
                for nm, fs in SETS.items():
                    v, pe = cvr2(sub, fs, ycol); row['d_' + nm] = v - base
                    if ycol == 'rm' and ps == 'all':
                        tt = sub.t.values; m2 = sub.pass2.values == 1
                        e0 = (yy[tt] - qq[tt]); eg = e0 - pb[:, None]; eb = e0 - pe[:, None]
                        f2 = lambda e: np.sqrt(np.mean(e[m2] ** 2)) if m2.any() else np.nan
                        oracle.append(dict(target=target, k=k, set=nm, n=len(sub), n2=int(m2.sum()),
                                           none=np.sqrt(np.mean(e0 ** 2)), gap=np.sqrt(np.mean(eg ** 2)), bridge=np.sqrt(np.mean(eb ** 2)),
                                           none_p2=f2(e0), gap_p2=f2(eg), bridge_p2=f2(eb)))
                out.append(row)
    for ycol in ['rm', 'r0']:
        sub = P[P.k >= 2].copy(); sub['kk'] = sub.k
        base, pb = cvr2(sub, GAP + ['kk'], ycol)
        row = dict(target=target, k='2-5pool', pass_='all', y=ycol, n=len(sub), sameid=sub.sameid.mean(), cert=sub.cert.mean(), r2_gap=base)
        for nm, fs in SETS.items():
            v, pe = cvr2(sub, fs + ['kk'], ycol); row['d_' + nm] = v - base
            if ycol == 'rm':
                tt = sub.t.values; m2 = sub.pass2.values == 1
                e0 = (yy[tt] - qq[tt]); eg = e0 - pb[:, None]; eb = e0 - pe[:, None]
                oracle.append(dict(target=target, k='2-5pool', set=nm, n=len(sub), n2=int(m2.sum()),
                                   none=np.sqrt(np.mean(e0 ** 2)), gap=np.sqrt(np.mean(eg ** 2)), bridge=np.sqrt(np.mean(eb ** 2)),
                                   none_p2=np.sqrt(np.mean(e0[m2] ** 2)), gap_p2=np.sqrt(np.mean(eg[m2] ** 2)), bridge_p2=np.sqrt(np.mean(eb[m2] ** 2))))
        out.append(row)
    P1 = P[P.k == 1]
    dy = (P1.dyT - P1.dya).values; dres = (P1.rm - P1.a_rm).values
    print(f'-- step3 {target}: n={len(P1)}  corr: [t-day level] with dy(day-mean change) / dres(residual change) | [t-a diff] with dy / dres')
    for c in DCOL:
        lv = D[c].values[P1.t.values]; df_ = P1['tfull_' + c].values
        cc = lambda u, v: np.corrcoef(u, v)[0, 1] if np.std(u) > 0 else np.nan
        print(f'   {c:7s} lvl: dy {cc(lv, dy):+.2f} dres {cc(lv, dres):+.2f} | diff: dy {cc(df_, dy):+.2f} dres {cc(df_, dres):+.2f}')
    print(f'-- step4 {target}: sameID fraction of path by k:', P.groupby('k').sameid.mean().round(3).to_dict(),
          ' all-cert path:', P.groupby('k').cert.mean().round(3).to_dict())
    P.to_csv(os.path.join(RES, 'agentX_bridge', f'hb1_pairs_{target}.csv'), index=False)

O = pd.DataFrame(out); OR = pd.DataFrame(oracle)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
print('\n### incremental CV R2 over gap-only (d_* = R2(set) - R2(gap))'); print(O.round(3).to_string())
print('\n### oracle hourly RMSE (true chain, constant day correction from CV pred of rm)'); print(OR.round(4).to_string())
O.to_csv(os.path.join(RES, 'agentX_bridge', 'hb1_r2.csv'), index=False); OR.to_csv(os.path.join(RES, 'agentX_bridge', 'hb1_oracle.csv'), index=False)
T2 = T.copy(); T2['p2'] = day[T2.b.values] >= 179
print('\ntruth a1 same_farm by pass:', T2.groupby('p2').same_farm.mean().round(3).to_dict(), ' a1 is test record frac:', round((~istrain[T2.a1.values]).mean(), 3))
