# -*- coding: utf-8 -*-
"""재분석 Q4·Q5 (2026-10-01, 집 클로드). 진단만.

Q5: G_C2 하루 오프셋(DIAG10)이 날 사이에 이어지는가 — 기록상 d-1, d-2, d-3 자기상관,
    같은 외기 24h 짝(같은 달력날의 다른 동) 사이 상관. 이어진다면 "숨은 상태"(입력으로 설명 안 되는 지속 요인).
Q4: 다른 49개 온실에서 따뜻한 행(in_temp>=12)으로 적합한 단순 지연 선형식이
    추운 행(in_temp<8)에서 과대/과소 예측하는가 — 온실별 부호 분포(외삽 편향의 사전 분포).
결과: local/re04_q4q5.txt
"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from harness import load

OUT = os.path.join(env.LOCAL, "re04_q4q5.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _f.write(s + "\n")


def gc2(lab, z, split):
    base = np.nanmean([z[f"{split}__MASK__7"], z[f"{split}__MASK__101"]], axis=0)
    cx = np.nanmean([z[f"{split}__CODEX__726"], z[f"{split}__CODEX__727"]], axis=0)
    pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{split}.npy")).mean(0)
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    return (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn


# ------------------------------------------------------------------ Q5
_, lab, _ = load()
z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
lab = lab.assign(pred=gc2(lab, z, "DIAG10"))
lab["e"] = lab.pred - lab.sub_temp
day = lab.groupby(["farm", "day"]).agg(e_day=("e", "mean")).reset_index()
p("Q5. DIAG10 하루 오프셋 자기상관 (같은 온실, 기록상 일차 간격 k)")
rng = np.random.default_rng(0)
for k in (1, 2, 3, 4, 7):
    a = day.merge(day.assign(day=day.day + k), on=["farm", "day"], suffixes=("", "_prev"))
    r = np.corrcoef(a.e_day, a.e_day_prev)[0, 1]
    null = [np.corrcoef(a.e_day, rng.permutation(a.e_day_prev.values))[0, 1] for _ in range(2000)]
    pv = (np.sum(np.abs(null) >= abs(r)) + 1) / 2001
    rf = {fm: round(np.corrcoef(*a.loc[a.farm == fm, ["e_day", "e_day_prev"]].values.T)[0, 1], 3) for fm in ("F13", "F47")}
    p(f"  k={k}: r {r:+.3f} (n {len(a)}, 순열 p {pv:.4f}) 온실별 {rf}")

# 같은 외기 24h 짝
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"] = X.row_id.str[:3]
X["day"] = X.row_id.str[4:7].astype(int)
X["hour"] = X.row_id.str[8:10].astype(int)
X = X.sort_values(["farm", "day", "hour"])
sig = X.groupby(["farm", "day"]).apply(lambda g: tuple(np.round(g[["out_temp", "out_hum", "out_rad", "out_wspd"]].values.ravel(), 1)) if len(g) == 24 else None)
sig = sig.dropna().reset_index()
sig.columns = ["farm", "day", "sig"]
pairs = sig.merge(sig, on="sig", suffixes=("_a", "_b"))
pairs = pairs[(pairs.farm_a + pairs.day_a.astype(str)) < (pairs.farm_b + pairs.day_b.astype(str))]
ed = day.set_index(["farm", "day"]).e_day
pairs["ea"] = [ed.get((f, d_), np.nan) for f, d_ in zip(pairs.farm_a, pairs.day_a)]
pairs["eb"] = [ed.get((f, d_), np.nan) for f, d_ in zip(pairs.farm_b, pairs.day_b)]
pairs = pairs.dropna(subset=["ea", "eb"])
for tag, m in [("같은 온실 짝", pairs.farm_a == pairs.farm_b), ("다른 온실 짝", pairs.farm_a != pairs.farm_b)]:
    q = pairs[m]
    if len(q) > 5:
        r = np.corrcoef(q.ea, q.eb)[0, 1]
        null = [np.corrcoef(q.ea, rng.permutation(q.eb.values))[0, 1] for _ in range(2000)]
        pv = (np.sum(np.abs(null) >= abs(r)) + 1) / 2001
        p(f"  같은 외기 24h {tag}: n {len(q)} r {r:+.3f} 순열 p {pv:.4f}")

# ------------------------------------------------------------------ Q4
p("\nQ4. 다른 온실: 따뜻한 행(in_temp>=12)으로 적합한 sub ~ in_temp 지연 0~3 선형식의 추운 행(<8) 편향")
Xa = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"])
Ya = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
A = Xa.merge(Ya, on="row_id")
A["farm"] = A.row_id.str[:3]
A["day"] = A.row_id.str[4:7].astype(int)
A["hour"] = A.row_id.str[8:10].astype(int)
A = A.sort_values(["farm", "day", "hour"])
A.loc[(A.in_temp < 0) | (A.in_temp > 45), "in_temp"] = np.nan
A.loc[A.sub_temp >= 40, "sub_temp"] = np.nan
rows = []
for fm, g in A.groupby("farm"):
    g = g.copy()
    for k in (1, 2, 3):
        g[f"l{k}"] = g.in_temp.shift(k).where(g.day.diff(k).fillna(99).abs() <= 1)
    g = g.dropna(subset=["in_temp", "l1", "l2", "l3", "sub_temp"])
    warm, cold = g[g.in_temp >= 12], g[g.in_temp < 8]
    if len(cold) < 30 or len(warm) < 500:
        continue
    Aw = np.c_[np.ones(len(warm)), warm[["in_temp", "l1", "l2", "l3"]].values]
    w = np.linalg.lstsq(Aw, warm.sub_temp.values, rcond=None)[0]
    Ac = np.c_[np.ones(len(cold)), cold[["in_temp", "l1", "l2", "l3"]].values]
    bias = float((Ac @ w - cold.sub_temp.values).mean())
    wres = float((Aw @ w - warm.sub_temp.values).std())
    rows.append((fm, len(cold), bias, wres, float((cold.sub_temp - cold.in_temp).mean())))
t = pd.DataFrame(rows, columns=["farm", "n_cold", "cold_bias", "warm_resid_sd", "cold_sub_minus_in"])
p(t.round(3).sort_values("cold_bias").to_string(index=False))
p(f"  온실 수 {len(t)}, 추운 행 과대예측(편향>0) {int((t.cold_bias>0).sum())}, 과소 {int((t.cold_bias<0).sum())}, "
  f"편향 중앙 {t.cold_bias.median():+.3f}")
_f.close()
