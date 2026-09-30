# -*- coding: utf-8 -*-
"""재분석 Q2 (2026-10-01, 집 클로드). 진단만.

Q2: G_C2의 하루 오프셋(날 평균 잔차)이 그날 입력과 관련 있는가?
 (1) 상한: 하루 전체 입력 요약 12개 vs 하루 오프셋 상관 (비인과, 신호 존재 여부만) + BH 보정
 (2) 인과: 시각 h까지의 누적 요약 vs 그 시각 잔차 e(h) — h가 커질수록 신호가 커지는가
 (3) 층화: 온실별, EXT10 vs DIAG10
부가: EXT10/EXT12 시각대 편향이 두 온실에서 같은 방향인가 (Q3 후속)
결과: local/re03_q2.txt
"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from harness import load

OUT = os.path.join(env.LOCAL, "re03_q2.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _f.write(s + "\n")


def rmse(e):
    return float(np.sqrt(np.nanmean(np.asarray(e) ** 2)))


def gc2(lab, z, split):
    base = np.nanmean([z[f"{split}__MASK__7"], z[f"{split}__MASK__101"]], axis=0)
    cx = np.nanmean([z[f"{split}__CODEX__726"], z[f"{split}__CODEX__727"]], axis=0)
    pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{split}.npy")).mean(0)
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    return (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn


def bh(pv):
    pv = np.asarray(pv)
    n = len(pv)
    o = np.argsort(pv)
    q = pv[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def perm_p(x, y, groups=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    r0 = abs(np.corrcoef(x, y)[0, 1])
    cnt = 0
    for _ in range(n):
        cnt += abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= r0
    return (cnt + 1) / (n + 1)


_, lab0, _ = load()
z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
lab = lab0.copy()
for s in ("EXT10", "EXT12", "DIAG10"):
    lab["pred_" + s] = gc2(lab0, z, s)
lab = lab.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
lab["dayid"] = lab.farm + "_" + lab.day.astype(str)
lab["dio"] = lab.in_temp - lab.out_temp
lab["dh"] = lab.in_temp - lab.groupby("dayid").in_temp.transform("first")

# 자정 점프: 이 날 0시 in_temp - 기록상 직전 행(전날 23시) in_temp
prev23 = lab.groupby("farm").in_temp.shift(1)
lab["mid_jump"] = np.where(lab.hour == 0, lab.in_temp - prev23, np.nan)
lab["mid_jump"] = lab.groupby("dayid").mid_jump.transform("first")

feats = {
    "in_temp_mean": ("in_temp", "mean"), "in_temp_min": ("in_temp", "min"), "in_temp_max": ("in_temp", "max"),
    "out_temp_mean": ("out_temp", "mean"), "dio_mean": ("dio", "mean"), "in_hum_mean": ("in_hum", "mean"),
    "in_co2_mean": ("in_co2", "mean"), "rad_sum": ("out_rad", "sum"), "heat_mean": ("act_heating", "mean"),
    "vent_mean": ("act_vent", "mean"), "thermal_mean": ("act_thermal", "mean"), "circfan_mean": ("act_circfan", "mean"),
}

for split in ("DIAG10", "EXT10"):
    col = "pred_" + split
    d = lab[lab[col].notna()].copy()
    d["e"] = d[col] - d.sub_temp
    day = d.groupby("dayid").agg(e_day=("e", "mean"), farm=("farm", "first"), dayn=("day", "first"),
                                 mid_jump=("mid_jump", "first"), **feats)
    p("=" * 100)
    p(f"[{split}] 날 {len(day)}  하루 오프셋 sd {day.e_day.std():.3f}")
    p(" (1) 하루 전체 요약 vs 하루 오프셋 (비인과 상한, 순열 p, BH q) — 온실 내 중심화 후")
    rows = []
    cols = list(feats) + ["mid_jump", "dayn"]
    for c in cols:
        x = day[c] - day.groupby("farm")[c].transform("mean")
        y = day.e_day - day.groupby("farm").e_day.transform("mean")
        m = pd.concat([x, y], axis=1).dropna()
        r = np.corrcoef(m.iloc[:, 0], m.iloc[:, 1])[0, 1]
        rf = [np.corrcoef(*day.loc[day.farm == fm, [c, "e_day"]].dropna().values.T)[0, 1] for fm in ("F13", "F47")]
        rows.append((c, r, rf[0], rf[1], perm_p(m.iloc[:, 0].values, m.iloc[:, 1].values, n=1000)))
    t = pd.DataFrame(rows, columns=["요약", "r", "r_F13", "r_F47", "p"])
    t["q_BH"] = bh(t.p)
    p(t.round(3).to_string(index=False))
    # 다변량 상한: 요약 전체로 하루 오프셋 설명 (날 단위 10겹 교차적합 릿지)
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    X = day[cols].copy()
    X = X.fillna(X.median())
    X = (X - X.mean()) / X.std()
    X["farm"] = (day.farm == "F47").astype(float)
    y = day.e_day.values
    pr = np.zeros(len(y))
    for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
        mdl = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X.iloc[tr], y[tr])
        pr[te] = mdl.predict(X.iloc[te])
    p(f"   다변량 교차적합 R² (하루 전체, 비인과): {1 - ((y-pr)**2).sum()/((y-y.mean())**2).sum():+.3f}")

    # (2) 인과: 시각 h까지 누적 평균 요약으로 e(h) 설명력 — h 구간별
    g = d.groupby("dayid")
    cum = pd.DataFrame(index=d.index)
    for c in ["in_temp", "out_temp", "dio", "in_hum", "in_co2", "act_heating", "act_vent", "act_thermal"]:
        cum["cm_" + c] = g[c].cumsum() / (g.cumcount() + 1)
    cum["mid_jump"] = d.mid_jump
    cum["farm"] = (d.farm == "F47").astype(float)
    cum = cum.fillna(cum.median())
    days = d.dayid.unique()
    fold = dict(zip(days, np.random.default_rng(1).integers(0, 10, len(days))))
    fk = d.dayid.map(fold).values
    p(" (2) 인과 누적요약 → 잔차 e(h), 날 10겹 교차적합 R² (시각 구간별 따로 적합)")
    for h0, h1 in [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23)]:
        mm = d.hour.between(h0, h1).values
        Xh = cum[mm].values
        Xh = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-9)
        yh = d.e.values[mm]
        pr = np.zeros(mm.sum())
        for k in range(10):
            tr, te = fk[mm] != k, fk[mm] == k
            mdl = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(Xh[tr], yh[tr])
            pr[te] = mdl.predict(Xh[te])
        r2 = 1 - ((yh - pr) ** 2).sum() / ((yh - yh.mean()) ** 2).sum()
        p(f"   {h0:2d}~{h1:2d}시: R² {r2:+.3f}  (RMSE {rmse(yh):.3f} → {rmse(yh-pr):.3f})")

# 부가: 시각대 편향의 온실 일관성
p("=" * 100)
p("시각대 편향 (3시간 구간, 온실별)")
for split in ("EXT10", "EXT12", "DIAG10"):
    col = "pred_" + split
    d = lab[lab[col].notna()].copy()
    d["e"] = d[col] - d.sub_temp
    tb = d.groupby([d.hour // 3 * 3, "farm"]).e.mean().unstack().round(3)
    p(f"[{split}]")
    p(tb.T.to_string())
_f.close()
