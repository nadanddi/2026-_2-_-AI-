# -*- coding: utf-8 -*-
"""ITP1 (2026-10-16 집 클로드; 사용자 '보간은 기상뿐 아니라 다른 부분에도 있지 않아?').  서술(판정 없음), 실행 전 커밋.
F13·F47 학습·평가 기록마다 열별로:
  직선 구간(보간 흔적) = 연속 >= 4시간 동안 이계차분 |Δ²x| <= tol 이면서 값이 변함(일차차분 |Δx| > tol) — 기울기 일정한 직선,
  멈춤 구간 = 연속 >= 6시간 같은 값(구동기는 0/100 등 운영상 일정이 흔하므로 따로 표시만).
  tol = 열 전체 일차차분 절댓값 중앙의 2% (정수형 값 열은 0.01).  정답 sub_temp·sub_ec(학습만)도 같은 검사.
출력: 열 x 집합(학습/평가)별 '직선 구간 있는 기록 수'·'직선 구간 행 수', 기록 목록(상위), 학습 큰 오차 기록과 겹침.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u itp1_interpolation_scan_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")).merge(pd.read_csv(os.path.join(env.DATA, "train_y.csv"))[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
def prep(X):
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:10].astype(int); return X.sort_values(["rec", "h"])
TR, TE = prep(TR), prep(TE)
def runs(mask, L):
    out = np.zeros(len(mask), bool); cnt = 0
    for i, m in enumerate(mask):
        cnt = cnt + 1 if m else 0
        if cnt >= L: out[i - L + 1:i + 1] = True
    return out
def scan(X, cols, tag):
    rows, recs = [], {}
    for c in cols:
        if c not in X: continue
        v_all = X[c].values; d1 = np.abs(np.diff(v_all)); tol = max(np.nanmedian(d1[d1 > 0]) * .02, 1e-6) if (d1 > 0).any() else 1e-6
        nlin = 0; nflat = 0; rl, rf = [], []
        for r, g in X.groupby("rec"):
            v = g[c].values.astype(float)
            if np.isfinite(v).sum() < 6: continue
            d = np.diff(v); dd = np.diff(d)
            lin_pt = np.zeros(len(v), bool); flat_pt = np.zeros(len(v), bool)
            m2 = np.isfinite(dd) & (np.abs(dd) <= tol) & (np.abs(d[1:]) > tol)          # straight non-flat segment at interior point
            lin_pt[1:-1] = runs(m2, 3)                                                    # 3 consecutive interior points = >= 4..5 hours straight
            mf = np.isfinite(d) & (np.abs(d) <= 1e-9); flat_pt[1:] = runs(mf, 5)          # 6 hours identical
            if lin_pt.sum(): nlin += 1; rl.append(r)
            if flat_pt.sum(): nflat += 1; rf.append(r)
            recs.setdefault(r, set()).update({c} if lin_pt.sum() else set())
        rows.append(dict(set=tag, col=c, recs_linear=nlin, recs_flat6h=nflat, linear_list=",".join(sorted(rl)[:12])))
    return pd.DataFrame(rows), recs
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
A, ra = scan(TR, COLS + ["sub_temp", "sub_ec"], "train(400)"); Bt, rb = scan(TE, COLS, "eval(60)")
print(pd.concat([A, Bt]).to_string(index=False))
big = ["F13_107", "F13_109", "F13_139", "F13_157", "F13_175", "F13_217", "F13_244", "F47_018", "F47_095", "F47_096", "F47_105", "F47_123", "F47_132", "F47_145", "F47_178", "F47_191", "F47_214", "F47_239", "F47_240", "F47_241"]
print("\nlarge day-level-error records (DHM1) with any straight-line input segment:", {r: sorted(ra.get(r, set())) for r in big if ra.get(r)})
print("eval records with any straight-line input segment:", {r: sorted(s) for r, s in rb.items() if s})
