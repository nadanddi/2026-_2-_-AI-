# -*- coding: utf-8 -*-
"""PI1c: PI1 2절 사건 반응을 대조 풀(같은 농장, 1·2차 통합)로 재계산 - 2차 대조 부족 해소."""

import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
OUT = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)

X = pd.read_csv(os.path.join(env.DATA, 'train_X.csv'))
Y = pd.read_csv(os.path.join(env.DATA, 'train_y.csv'))
X = X[X.row_id.str[:3].isin(['F13', 'F47'])].merge(Y, on='row_id')
X['farm'] = X.row_id.str[:3]; X['day'] = X.row_id.str[4:7].astype(int); X['hour'] = X.row_id.str[-2:].astype(int)
X['rec'] = X.farm + '_' + X.row_id.str[4:7]
X['ph'] = np.where(X.day < 179, 1, 2)
X['EC25'] = X.sub_ec / (1 + 0.02 * (X.sub_temp - 25))
ACT = ['act_vent', 'act_thermal', 'act_shade', 'act_heating', 'act_circfan', 'act_co2', 'act_fog']
V = ['out_temp', 'out_hum', 'out_rad', 'out_wspd', 'in_temp', 'in_hum', 'in_co2'] + ACT + ['sub_temp', 'EC25']
X = X.sort_values(['rec', 'hour']).reset_index(drop=True)
recs = X.rec.unique(); R = len(recs)
assert (X.groupby('rec').size() == 24).all()
A = X[V].to_numpy(float).reshape(R, 24, len(V))
recinfo = X.groupby('rec', sort=False)[['farm', 'ph', 'day']].first().loc[recs].reset_index()
cell = (recinfo.farm + '_' + recinfo.ph.astype(str)).to_numpy()
vi = {v: i for i, v in enumerate(V)}

D = np.full_like(A, np.nan); D[:, 1:] = A[:, 1:] - A[:, :-1]  # change into hour t

def lag(arr, L):
    o = np.full_like(arr, np.nan)
    if L == 0: return arr.copy()
    o[:, L:] = arr[:, :-L]; return o

HOUR = np.tile(np.arange(24), (R, 1))
RECI = np.tile(np.arange(R)[:, None], (1, 24))

def partial(y, x, ctrls, mask_rec):
    """y,x,(ctrls list) arrays RxT; returns pr, beta, cluster z, n"""
    m = mask_rec[:, None] & np.isfinite(y) & np.isfinite(x)
    for c in ctrls: m &= np.isfinite(c)
    m[:, :6] = False  # need lags up to 5 inside record -> hours>=6 only for comparable sample
    if m.sum() < 200: return np.nan, np.nan, np.nan, 0
    yy = y[m]; xx = x[m]; h = HOUR[m]; g = RECI[m]
    Z = [np.ones_like(yy)] + [c[m] for c in ctrls] + [(h == k).astype(float) for k in range(7, 24)]
    Z = np.column_stack(Z)
    q, _ = np.linalg.qr(Z)
    yr = yy - q @ (q.T @ yy); xr = xx - q @ (q.T @ xx)
    sx = (xr ** 2).sum()
    if sx < 1e-9 or (yr ** 2).sum() < 1e-9: return np.nan, np.nan, np.nan, int(m.sum())
    pr = (xr * yr).sum() / np.sqrt(sx * (yr ** 2).sum())
    b = (xr * yr).sum() / sx; e = yr - b * xr
    s = pd.Series(xr * e).groupby(g).sum().to_numpy()
    se = np.sqrt((s ** 2).sum()) / sx
    return pr, b, b / se, int(m.sum())

def rawcorr(y, x, mask_rec):
    m = mask_rec[:, None] & np.isfinite(y) & np.isfinite(x); m[:, :6] = False
    if m.sum() < 200: return np.nan
    a, b = y[m], x[m]
    if a.std() == 0 or b.std() == 0: return np.nan
    return np.corrcoef(a, b)[0, 1]

# ---- 2. event responses (pooled control)

print('=' * 30, '2. 계단 사건 반응', flush=True)
RESP = ['in_temp', 'in_hum', 'in_co2', 'sub_temp', 'EC25']
ACTD = {'act_vent': (5, 1), 'act_thermal': (20, 10), 'act_shade': (20, 10), 'act_heating': (10, 1),
        'act_circfan': (10, 1), 'act_co2': (10, 1), 'act_fog': (5, 1)}
OT = A[:, :, vi['out_temp']]; ORD = A[:, :, vi['out_rad']]
farmarr = (recinfo.farm == 'F13').to_numpy().astype(int)
def events(act, up=True):
    a = A[:, :, vi[act]]; thr_d, lo = ACTD[act]
    ev = []
    for r in range(R):
        for t in range(2, 19):
            p, c = a[r, t - 1], a[r, t]
            if not (np.isfinite(p) and np.isfinite(c)): continue
            if up and c - p >= thr_d and p < lo: ev.append((r, t))
            if (not up) and p - c >= thr_d and c < lo: ev.append((r, t))
    return ev
evrows = []
for act in ACT:
    for up in ([True, False] if act in ('act_thermal', 'act_shade', 'act_vent', 'act_heating') else [True]):
        ev = events(act, up)
        a = A[:, :, vi[act]]
        evset = set(ev)
        for ph in (1, 2):
            evp = [(r, t) for r, t in ev if recinfo.ph[r] == ph]
            if len(evp) < 5:
                evrows.append(dict(act=act, dir='on' if up else 'off', ph=ph, n=len(evp))); continue
            diffs = {v: [] for v in RESP}; recs_ev = []
            for r, t in evp:
                # control: same farm & phase, same hour t, actuator steady (|change|<thr over t-2..t+5 window), nearest out_temp/out_rad at t
                ok = (farmarr == farmarr[r])  # PI1c: 대조 풀 = 같은 농장 1·2차 전체
                win = a[:, max(t - 2, 0):min(t + 6, 24)]
                steady = np.nanmax(win, 1) - np.nanmin(win, 1) < ACTD[act][0]
                cand = np.where(ok & steady & (np.arange(R) != r))[0]
                if len(cand) < 10: continue
                dist = ((OT[cand, t] - OT[r, t]) / 5) ** 2 + ((ORD[cand, t] - ORD[r, t]) / 200) ** 2 + ((OT[cand, t - 1] - OT[r, t - 1]) / 5) ** 2
                dist = np.where(np.isfinite(dist), dist, np.inf)
                nn = cand[np.argsort(dist)[:15]]
                recs_ev.append(r)
                for v in RESP:
                    s = A[:, :, vi[v]]
                    idx = np.arange(t - 2, t + 6); valid = idx < 24
                    tr = np.full(8, np.nan); ct = np.full(8, np.nan)
                    tr[valid] = s[r, idx[valid]] - s[r, t - 1]
                    ct[valid] = np.nanmean(s[nn][:, idx[valid]] - s[nn, t - 1][:, None], 0)
                    diffs[v].append(tr - ct)
            recs_ev = np.array(recs_ev)
            if len(recs_ev) < 5:
                evrows.append(dict(act=act, dir='on' if up else 'off', ph=ph, n=len(recs_ev))); continue
            out = dict(act=act, dir='on' if up else 'off', ph=ph, n=len(recs_ev), nrec=len(np.unique(recs_ev)),
                       hour_med=np.median([t for r, t in evp]))
            for v in RESP:
                M = np.array(diffs[v])
                mean = np.nanmean(M, 0)
                # record bootstrap CI for +2h (index 4 = t+2? idx: t-2..t+5 -> pos 0..7; t+1=pos3,t+2=pos4,t+3=pos5,t+5=pos7)
                ur = np.unique(recs_ev); bs = []
                for _ in range(300):
                    pick = rng.choice(ur, len(ur))
                    rows_ = np.concatenate([np.where(recs_ev == q)[0] for q in pick])
                    bs.append(np.nanmean(M[rows_], 0))
                bs = np.array(bs); lo_, hi_ = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
                for k, lab in [(0, '-2'), (2, '0'), (3, '+1'), (4, '+2'), (5, '+3'), (7, '+5')]:
                    out[f'{v}@{lab}'] = mean[k]
                for k, lab in [(3, '+1'), (5, '+3')]:
                    out[f'{v}@{lab}_sig'] = int(lo_[k] > 0) - int(hi_[k] < 0)
            evrows.append(out)
EV = pd.DataFrame(evrows)
EV.to_csv(os.path.join(OUT, 'pi1c_events_pooledctrl_v1.csv'), index=False)
for v in RESP:
    cols = ['act', 'dir', 'ph', 'n', 'nrec', 'hour_med'] + [c for c in EV.columns if c.startswith(v + '@')]
    print('-- 반응:', v); print(EV[cols].round(3).to_string(index=False))

