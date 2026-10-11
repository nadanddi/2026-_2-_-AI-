# -*- coding: utf-8 -*-
"""H-CONTEMP 1단계: 다른 온실이 F13·F47과 같은 장소·같은 날짜인지 (학습 기록만, 서술).

다른 48개 온실(F32 제외)은 외기(out_*)가 전부 결측 -> 외기 곡선 직접 일치 불가.
대신 내부 일사(in_rad, 다른 온실 98%) 하루 합/시간 곡선과 F13·F47 외부 일사(out_rad)
하루 합/시간 곡선을 달력(cal_own_farm_v1 'cal') 위에서 지연(lag) 탐색으로 맞춘다.
다른 온실 기록 일차 = 연속 달력 일이라고 가정(이어붙임은 F13·F47만, 카탈로그 1.3).
F32는 out_temp/out_rad가 있으므로 같은 방식 + out_temp로도 맞춤.
산출: local/hc1_lag_scan_v1.csv, 로그는 stdout.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import env
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local'); os.makedirs(OUT, exist_ok=True)
X = pd.read_csv(os.path.join(env.DATA, 'train_X.csv'))
X['farm'] = X.row_id.str[:3]; X['day'] = X.row_id.str[4:7].astype(int); X['hour'] = X.row_id.str[8:10].astype(int)
cal = pd.read_csv(os.path.join(env.HERE, 'local', 'cal_own_farm_v1.csv'))[['farm', 'day', 'cal']]

# --- 기준: F13·F47 학습 기록, 달력 날짜별 out_rad 시간 곡선(평균) ---
T = X[X.farm.isin(['F13', 'F47'])].merge(cal, on=['farm', 'day'])
ref_h = T.pivot_table(index='cal', columns='hour', values='out_rad', aggfunc='mean')
ref_t = T.pivot_table(index='cal', columns='hour', values='out_temp', aggfunc='mean')
ref_rs = ref_h.sum(1); ref_tm = ref_t.mean(1)
print('기준 달력 날짜 수(학습):', len(ref_rs), 'cal 범위', ref_rs.index.min(), ref_rs.index.max())

def detr(s, w=15):
    s = s.sort_index()
    full = s.reindex(range(s.index.min(), s.index.max() + 1))
    tr = full.rolling(w, center=True, min_periods=5).median()
    return (full - tr).reindex(s.index)

ref_rs_d = detr(ref_rs); ref_tm_d = detr(ref_tm)
rows = []
farms = sorted(X.farm.unique())
for f in farms:
    if f in ('F13', 'F47'):
        continue
    D = X[X.farm == f]
    var = 'in_rad' if D.in_rad.notna().mean() > 0.5 else 'out_rad'
    oh = D.pivot_table(index='day', columns='hour', values=var, aggfunc='mean')
    ors = oh.sum(1, min_count=20); ors_d = detr(ors)
    otm_d = detr(D.groupby('day').out_temp.mean()) if D.out_temp.notna().mean() > .5 else None
    res = []
    for L in range(-400, 401):
        k = ref_rs_d.index + L
        a = ref_rs_d.values; b = ors_d.reindex(k).values
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 30:
            continue
        r = np.corrcoef(a[ok], b[ok])[0, 1]
        # 시간 곡선 정규화 거리(모양): 각 날 합으로 나눈 곡선의 상관
        res.append((L, ok.sum(), r))
    if not res:
        print(f, var, '겹침 없음'); continue
    R = pd.DataFrame(res, columns=['lag', 'n', 'r'])
    i = R.r.idxmax(); best = R.loc[i]
    others = R[(R.lag - best.lag).abs() > 3].r
    z = (best.r - others.mean()) / others.std()
    # 최적 지연에서 시간 곡선 모양 상관(정규화)
    k = ref_h.index + int(best.lag)
    A = ref_h.div(ref_h.sum(1), axis=0); B = oh.div(oh.sum(1), axis=0).reindex(k); B.index = A.index
    ok = A.notna().all(1) & B.notna().all(1)
    shape_r = np.corrcoef(A[ok].values.ravel(), B[ok].values.ravel())[0, 1] if ok.sum() else np.nan
    rt = np.nan
    if otm_d is not None:
        bb = otm_d.reindex(ref_tm_d.index + int(best.lag)).values; aa = ref_tm_d.values
        ok2 = np.isfinite(aa) & np.isfinite(bb); rt = np.corrcoef(aa[ok2], bb[ok2])[0, 1]
    rows.append(dict(farm=f, var=var, best_lag=int(best.lag), n=int(best.n), r_best=best.r,
                     r_2nd=others.max(), z=z, shape_r=shape_r, r_outtemp=rt, ndays=len(ors)))
S = pd.DataFrame(rows).sort_values('r_best', ascending=False)
S.to_csv(os.path.join(OUT, 'hc1_lag_scan_v1.csv'), index=False)
pd.set_option('display.width', 200)
print(S.round(3).to_string(index=False))
