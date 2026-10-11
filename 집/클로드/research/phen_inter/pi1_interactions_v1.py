# -*- coding: utf-8 -*-
"""PI1 온실 변수 상호작용 서술 분석 (F13·F47, 같은 기록 안 시차만). 서술 전용, 모델 변경 없음.
1) 시차 0~4 상관 + 편상관(대상 자기시차1·2, 외기온 시차0~2, 일사 시차0~3, 시각 고정효과) 기록 군집 SE
2) 구동기 계단 사건 반응(−2~+5h, 같은 농장·시각·외기 kNN 대조 차감)
3) 2x2 상호작용
5) 기록별 반응 크기 vs EC 수준·모델 잔차
"""
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

# ---------------- 1. lag matrix
print('=' * 30, '1. 시차 영향 행렬', flush=True)
rows = []
cells = ['F13_1', 'F47_1', 'F13_2', 'F47_2']
phmask = {1: recinfo.ph.to_numpy() == 1, 2: recinfo.ph.to_numpy() == 2}
cellmask = {c: cell == c for c in cells}
for tgt in V:
    y = D[:, :, vi[tgt]]
    for src in V:
        if src == tgt: continue
        x0 = D[:, :, vi[src]]
        for L in range(0, 5):
            x = lag(x0, L)
            ctr = [lag(y, 1), lag(y, 2)]
            if src != 'out_temp' and tgt != 'out_temp': ctr += [lag(D[:, :, vi['out_temp']], k) for k in range(3)]
            if src != 'out_rad' and tgt != 'out_rad': ctr += [lag(D[:, :, vi['out_rad']], k) for k in range(4)]
            rec = dict(src=src, tgt=tgt, L=L)
            for p in (1, 2):
                rec[f'r{p}'] = rawcorr(y, x, phmask[p])
                pr, b, z, n = partial(y, x, ctr, phmask[p])
                rec[f'pr{p}'] = pr; rec[f'b{p}'] = b; rec[f'z{p}'] = z; rec[f'n{p}'] = n
            for c in cells:
                rec['pr_' + c] = partial(y, x, ctr, cellmask[c])[0]
            rows.append(rec)
    print(tgt, flush=True)
L1 = pd.DataFrame(rows)
L1.to_csv(os.path.join(OUT, 'pi1_lagmatrix_v1.csv'), index=False)
cc = ['pr_' + c for c in cells]
L1['same4'] = (np.sign(L1[cc]).nunique(axis=1) == 1) & L1[cc].notna().all(axis=1)
L1['minabs'] = L1[cc].abs().min(axis=1)
# placebo floor: indoor/actuator/sub -> outdoor
inside = ['in_temp', 'in_hum', 'in_co2'] + ACT + ['sub_temp', 'EC25']
plc = L1[L1.src.isin(inside) & L1.tgt.isin(['out_temp', 'out_hum', 'out_rad', 'out_wspd']) & (L1.L >= 1)]
print('위약(실내→외기, 시차1~4) |pr| 분포 1차 95%/최대', plc.pr1.abs().quantile(.95).round(3), plc.pr1.abs().max().round(3),
      ' 2차', plc.pr2.abs().quantile(.95).round(3), plc.pr2.abs().max().round(3), ' |z| 95%', plc[['z1', 'z2']].abs().stack().quantile(.95).round(2),
      ' same4 비율', plc.same4.mean().round(3))
thr = max(.06, plc[['pr1', 'pr2']].abs().stack().quantile(.95))
sig = L1[L1.same4 & (L1.pr1.abs() >= thr) & (L1.pr2.abs() >= thr) & (L1.z1.abs() > 3) & (L1.z2.abs() > 3)].copy()
# keep best lag per pair (>=1), plus L0
best = sig[sig.L >= 1].assign(a=lambda d: (d.pr1.abs() + d.pr2.abs()) / 2).sort_values('a', ascending=False).drop_duplicates(['src', 'tgt'])
print('임계 |pr| >=', round(thr, 3))
print('-- 시차>=1 유의 화살표(쌍별 최강 시차)')
print(best[['src', 'tgt', 'L', 'r1', 'pr1', 'z1', 'r2', 'pr2', 'z2', 'minabs']].round(3).to_string(index=False))
s0 = sig[sig.L == 0].assign(a=lambda d: (d.pr1.abs() + d.pr2.abs()) / 2).sort_values('a', ascending=False)
s0 = s0[s0.src < s0.tgt]
print('-- 동시(시차0) 유의 쌍(방향 미정)')
print(s0[['src', 'tgt', 'pr1', 'pr2', 'minabs']].round(3).to_string(index=False))
# EC25 & sub_temp targets full view
for t in ['EC25', 'sub_temp']:
    sub = L1[(L1.tgt == t)].assign(a=lambda d: (d.pr1.abs() + d.pr2.abs()) / 2).sort_values('a', ascending=False).head(12)
    print(f'-- 대상 {t} 상위 12(일관성 무관)'); print(sub[['src', 'L', 'r1', 'pr1', 'z1', 'r2', 'pr2', 'z2', 'same4']].round(3).to_string(index=False))

# ---------------- 2. event responses
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
                ok = (farmarr == farmarr[r]) & (recinfo.ph.to_numpy() == ph)
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
EV.to_csv(os.path.join(OUT, 'pi1_events_v1.csv'), index=False)
for v in RESP:
    cols = ['act', 'dir', 'ph', 'n', 'nrec', 'hour_med'] + [c for c in EV.columns if c.startswith(v + '@')]
    print('-- 반응:', v); print(EV[cols].round(3).to_string(index=False))

# ---------------- 3. 2x2 interactions
print('=' * 30, '3. 2x2 상호작용', flush=True)
F = X.copy()
for v in ['in_temp', 'sub_temp', 'EC25', 'in_hum', 'in_co2']:
    F['d2_' + v] = F.groupby('rec')[v].shift(-2) - F[v]   # 2시간 뒤 변화(서술용, 미래 사용)
F['gap_in_out'] = F.in_temp - F.out_temp
F['gap_sub_in'] = F.sub_temp - F.in_temp
OUTS = ['gap_in_out', 'in_hum', 'in_co2', 'gap_sub_in', 'd2_in_temp', 'd2_sub_temp', 'd2_EC25']
# residualize outcomes on farm*phase*hour means + out_temp + out_rad (linear within farm*phase)
for o in OUTS:
    F[o + '_r'] = np.nan
    for (fm, ph), g in F.groupby(['farm', 'ph']):
        gg = g[[o, 'out_temp', 'out_rad', 'hour']].dropna()
        Z = np.column_stack([pd.get_dummies(gg.hour).to_numpy(float), gg.out_temp, gg.out_rad])
        b = np.linalg.lstsq(Z, gg[o].to_numpy(), rcond=None)[0]
        F.loc[gg.index, o + '_r'] = gg[o] - Z @ b
PAIRS = [('보온커튼×난방', F.act_thermal >= 50, F.act_heating > 0),
         ('차광×천창', F.act_shade >= 50, F.act_vent > 0),
         ('밀폐(천창0)×유동팬', F.act_vent == 0, F.act_circfan > 0),
         ('CO2공급×천창', F.act_co2 > 0, F.act_vent > 0)]
irows = []
for name, a_, b_ in PAIRS:
    for ph in (1, 2):
        sel = (F.ph == ph) & F[['act_thermal', 'act_vent']].notna().all(1)
        for o in OUTS:
            cellm = {}
            for ai in (0, 1):
                for bi in (0, 1):
                    m = sel & (a_ == bool(ai)) & (b_ == bool(bi))
                    cellm[(ai, bi)] = (F.loc[m, o + '_r'].mean(), int(m.sum()))
            inter = cellm[(1, 1)][0] - cellm[(1, 0)][0] - cellm[(0, 1)][0] + cellm[(0, 0)][0]
            # record-bootstrap of interaction
            g = F[sel].copy(); g['A'] = a_[sel].astype(int); g['B'] = b_[sel].astype(int)
            g = g[['rec', 'A', 'B', o + '_r']].dropna()
            agg = g.groupby(['rec', 'A', 'B'])[o + '_r'].agg(['sum', 'count']).reset_index()
            ur = agg.rec.unique(); bs = []
            piv_s = agg.pivot_table(index='rec', columns=['A', 'B'], values='sum', fill_value=0)
            piv_c = agg.pivot_table(index='rec', columns=['A', 'B'], values='count', fill_value=0)
            for _ in range(300):
                pk = rng.integers(0, len(piv_s), len(piv_s))
                S = piv_s.iloc[pk].sum(); C = piv_c.iloc[pk].sum()
                mm = S / C.replace(0, np.nan)
                try: bs.append(mm[(1, 1)] - mm[(1, 0)] - mm[(0, 1)] + mm[(0, 0)])
                except KeyError: pass
            lo_, hi_ = (np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)) if len(bs) > 50 else (np.nan, np.nan)
            irows.append(dict(pair=name, ph=ph, out=o, m00=cellm[(0, 0)][0], m10=cellm[(1, 0)][0], m01=cellm[(0, 1)][0], m11=cellm[(1, 1)][0],
                              n00=cellm[(0, 0)][1], n10=cellm[(1, 0)][1], n01=cellm[(0, 1)][1], n11=cellm[(1, 1)][1],
                              inter=inter, lo=lo_, hi=hi_))
IR = pd.DataFrame(irows); IR.to_csv(os.path.join(OUT, 'pi1_interact2x2_v1.csv'), index=False)
print('(값 = 농장·차수·시각 평균과 외기온·일사 선형 효과를 뺀 잔차 평균; A=첫째 on, B=둘째 on; inter=11-10-01+00, 기록 부트스트랩 95%)')
print(IR.round(3).to_string(index=False))

# ---------------- 5. per-record response & EC / residual
print('=' * 30, '5. 기록별 반응 크기', flush=True)
P = pd.read_csv(r'C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv')
P = P[(P.tag == 'CUR') & (P.validator == 'DIAG10')]
P = P.groupby('row_id', as_index=False)[['sg', 'sub_ec']].mean()
P['res'] = P.sg - P.sub_ec; P['rec'] = P.row_id.str[:7]; P['hour'] = P.row_id.str[-2:].astype(int)
recres = P.groupby('rec').res.mean().rename('res_mean').to_frame()
recres['res_0'] = P[P.hour == 0].set_index('rec').res
recres['absres'] = P.groupby('rec').res.apply(lambda s: np.sqrt((s ** 2).mean()))
print('잔차 있는 기록 수', len(recres))
feat = []
for r in range(R):
    din = D[r, :, vi['in_temp']]; dsub = D[r, :, vi['sub_temp']]
    tin = A[r, :, vi['in_temp']]; tsub = A[r, :, vi['sub_temp']]
    rowsd = {}
    # coupling: dsub_t ~ din_{t-1..t-3} sum of coefs
    Zl = np.column_stack([np.r_[np.full(k, np.nan), din[:-k]] for k in (1, 2, 3)])
    m = np.isfinite(dsub) & np.isfinite(Zl).all(1)
    if m.sum() >= 12:
        Z = np.column_stack([np.ones(m.sum()), Zl[m]]); b = np.linalg.lstsq(Z, dsub[m], rcond=None)[0]
        rowsd['k_lag'] = b[1:].sum()
    # level slope: sub_t on in_{t-2}
    x2 = np.r_[np.nan, np.nan, tin[:-2]]; m2 = np.isfinite(x2) & np.isfinite(tsub)
    if m2.sum() >= 12 and np.nanstd(x2[m2]) > .3:
        rowsd['k_lvl'] = np.polyfit(x2[m2], tsub[m2], 1)[0]
    rowsd['amp_ratio'] = np.nanstd(tsub) / max(np.nanstd(tin), .1)
    rowsd['night_gap'] = np.nanmean(tsub[:6] - tin[:6])
    # heating response: mean dsub in the 3h after heating on vs off hours (night 0-7, 18-23)
    h = A[r, :, vi['act_heating']]
    rowsd['heat_hours'] = np.nansum(h > 0)
    # EC25 daily range / ec level
    e = A[r, :, vi['EC25']]
    rowsd['ec25_mean'] = np.nanmean(e); rowsd['ec25_0'] = e[0]; rowsd['ec25_range'] = np.nanmax(e) - np.nanmin(e)
    rowsd['ec_raw_mean'] = np.nanmean(X.sub_ec.to_numpy().reshape(R, 24)[r])
    rowsd['dsub_from_ec_tempcorr'] = np.corrcoef(np.diff(tsub), np.diff(X.sub_ec.to_numpy().reshape(R, 24)[r]))[0, 1] if np.nanstd(np.diff(tsub)) > 0 else np.nan
    feat.append(rowsd)
FE = pd.DataFrame(feat); FE['rec'] = recs; FE['ph'] = recinfo.ph.values; FE['farm'] = recinfo.farm.values
FE = FE.merge(recres, left_on='rec', right_index=True, how='left')
FE.to_csv(os.path.join(OUT, 'pi1_record_response_v1.csv'), index=False)
print(FE.groupby(['farm', 'ph'])[['k_lag', 'k_lvl', 'amp_ratio', 'night_gap', 'ec25_mean']].describe().T.round(3).to_string())
from scipy.stats import spearmanr
def permp(a, b, grp, n=2000):
    ok = np.isfinite(a) & np.isfinite(b); a, b, grp = a[ok], b[ok], grp[ok]
    obs = spearmanr(a, b)[0]; c = 0
    for _ in range(n):
        bb = b.copy()
        for gv in np.unique(grp):
            ix = np.where(grp == gv)[0]; bb[ix] = rng.permutation(bb[ix])
        c += abs(spearmanr(a, bb)[0]) >= abs(obs)
    return obs, (c + 1) / (n + 1), ok.sum()
rr = []
for k in ['k_lag', 'k_lvl', 'amp_ratio', 'night_gap', 'heat_hours']:
    for t in ['ec25_mean', 'ec_raw_mean', 'ec25_range', 'res_mean', 'res_0', 'absres']:
        for ph in (1, 2):
            g = FE[FE.ph == ph]
            rho, p, n = permp(g[k].to_numpy(float), g[t].to_numpy(float), g.farm.to_numpy(), 1000)
            rr.append(dict(resp=k, target=t, ph=ph, rho=rho, p_perm=p, n=n))
RR = pd.DataFrame(rr); RR.to_csv(os.path.join(OUT, 'pi1_record_assoc_v1.csv'), index=False)
W = RR.pivot_table(index=['resp', 'target'], columns='ph', values=['rho', 'p_perm']).round(3)
print('(스피어만, 농장 안 순열 p, 30개 검정 × 2차수 → 본페로니 기준 p<.0017)')
print(W.to_string())
# farm-adjusted check of night_gap vs EC (within farm*phase)
print('done')
