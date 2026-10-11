# -*- coding: utf-8 -*-
"""H-CONTEMP 1단계 보강: (a) 양성 대조 - 다른 온실끼리 in_rad 하루 합(추세 제거) 지연 상관,
(b) F13·F47 out_rad 대 다른 온실 in_rad를 겹침 n>=60 으로 제한해 다시 탐색 (학습 기록만)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import env
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local')
X = pd.read_csv(os.path.join(env.DATA, 'train_X.csv'))
X['farm'] = X.row_id.str[:3]; X['day'] = X.row_id.str[4:7].astype(int); X['hour'] = X.row_id.str[8:10].astype(int)
cal = pd.read_csv(os.path.join(env.HERE, 'local', 'cal_own_farm_v1.csv'))[['farm', 'day', 'cal']]

def detr(s, w=15):
    s = s.sort_index()
    full = s.reindex(range(s.index.min(), s.index.max() + 1))
    tr = full.rolling(w, center=True, min_periods=5).median()
    return (full - tr).reindex(s.index)

def scan(a, b, lags, nmin):
    out = []
    for L in lags:
        bb = b.reindex(a.index + L).values; aa = a.values
        ok = np.isfinite(aa) & np.isfinite(bb)
        if ok.sum() >= nmin:
            out.append((L, ok.sum(), np.corrcoef(aa[ok], bb[ok])[0, 1]))
    return pd.DataFrame(out, columns=['lag', 'n', 'r'])

ser = {}
for f, D in X.groupby('farm'):
    if f in ('F13', 'F47'):
        continue
    v = 'in_rad' if D.in_rad.notna().mean() > .5 else 'out_rad'
    ser[f] = detr(D.pivot_table(index='day', columns='hour', values=v, aggfunc='mean').sum(axis=1, min_count=20))
# 기록 일차 범위
rng = pd.DataFrame({f: (s.index.min(), s.index.max()) for f, s in ser.items()}, index=['d0', 'd1']).T
print('다른 온실 기록 일차 범위 분포:\n', rng.describe().round(1).to_string())

# (a) 다른 온실끼리 lag -60..60
fs = sorted(ser)
best = []
for i, f in enumerate(fs):
    for g in fs[i + 1:]:
        R = scan(ser[f], ser[g], range(-60, 61), 80)
        if len(R) == 0:
            continue
        j = R.r.idxmax(); r0 = R[R.lag == 0].r
        best.append((f, g, int(R.lag[j]), R.r[j], float(r0.iloc[0]) if len(r0) else np.nan))
B = pd.DataFrame(best, columns=['f', 'g', 'lag', 'r', 'r_lag0'])
print('\n(a) 다른 온실 쌍', len(B), '| 최적 r 분포:', B.r.describe().round(3).to_dict())
print('최적 지연 = 0 비율', (B.lag == 0).mean().round(3), '| lag0 r 중앙', B.r_lag0.median().round(3))
print('최적 r>=0.8 쌍 수', (B.r >= .8).sum(), '| 상위 15쌍:')
print(B.sort_values('r', ascending=False).head(15).round(3).to_string(index=False))
B.to_csv(os.path.join(OUT, 'hc2_pairs_v1.csv'), index=False)

# (b) F13·F47 기준(out_rad), n>=60
T = X[X.farm.isin(['F13', 'F47'])].merge(cal, on=['farm', 'day'])
ref = detr(T.pivot_table(index='cal', columns='hour', values='out_rad', aggfunc='mean').sum(axis=1))
rows = []
for f in fs:
    R = scan(ref, ser[f], range(-400, 401), 60)
    if len(R) == 0:
        rows.append((f, np.nan, 0, np.nan, np.nan)); continue
    j = R.r.idxmax()
    rows.append((f, int(R.lag[j]), int(R.n[j]), R.r[j], R[(R.lag - R.lag[j]).abs() > 3].r.max()))
S = pd.DataFrame(rows, columns=['farm', 'lag', 'n', 'r_best', 'r_2nd']).sort_values('r_best', ascending=False)
print('\n(b) F13·F47 out_rad 대 다른 온실 (n>=60):')
print(S.round(3).to_string(index=False))
S.to_csv(os.path.join(OUT, 'hc2_ref_scan_v1.csv'), index=False)
