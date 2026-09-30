# -*- coding: utf-8 -*-
"""재분석 Q3 후속 (2026-10-01, 집 클로드). 진단만.

re02 발견: EXT10에서는 하루 안 잔차가 dT1(+0.22)·drad1(+0.27)과 상관, DIAG10에서는 ~0.
확인할 것
 (a) 시각 효과를 뺀 부분상관 (같은 시각끼리 중심화) — 아침 상승 시각 교란 배제
 (b) EXT12에서도 같은가
 (c) 같은 85개 추운 날을 DIAG10 예측으로 보면 신호가 있는가
     (있으면 물리적 성질, 없으면 학습 범위 밖 외삽의 산물)
 (d) 지연 모양: 같은 날 안에서 y ~ a*pred + b*pred(h-1) 을 날 블록 교차적합으로 추정
결과: local/re02b_q3.txt
"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from harness import load

OUT = os.path.join(env.LOCAL, "re02b_q3.txt")
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


_, lab0, _ = load()
z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
P = {s: gc2(lab0, z, s) for s in ("EXT10", "EXT12", "DIAG10")}
lab = lab0.copy()
for s in P:
    lab["pred_" + s] = P[s]
lab = lab.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
lab["dayid"] = lab.farm + "_" + lab.day.astype(str)
g = lab.groupby("dayid")
lab["dT1"] = g.in_temp.diff()
lab["drad1"] = g.out_rad.diff()


def analyse(d, col, tag):
    d = d[d[col].notna()].copy()
    d["e"] = d[col] - d.sub_temp
    d["e_in"] = d.e - d.groupby("dayid").e.transform("mean")
    gg = d.groupby("dayid")
    d["pl1"] = gg[col].shift(1)
    # (a) 시각 중심화 부분상관
    out = [f"[{tag}] 날 {d.dayid.nunique()} 행 {len(d)} RMSE {rmse(d.e):.4f} 하루안 RMSE {rmse(d.e_in):.4f}"]
    for c in ("dT1", "drad1"):
        m = d[[c, "e_in", "hour"]].dropna()
        xc = m[c] - m.groupby("hour")[c].transform("mean")
        yc = m.e_in - m.groupby("hour").e_in.transform("mean")
        out.append(f"   {c}: r {np.corrcoef(m[c], m.e_in)[0,1]:+.3f} | 시각중심화 r {np.corrcoef(xc, yc)[0,1]:+.3f}")
    # (d) 지연 혼합: 날 단위 5겹 교차적합, y_in ~ a*pred_in + b*pl1_in (하루 평균 제거 후, 하루 안 모양만)
    m = d[["dayid", "sub_temp", col, "pl1"]].dropna().copy()
    for c in ("sub_temp", col, "pl1"):
        m[c + "_c"] = m[c] - m.groupby("dayid")[c].transform("mean")
    days = m.dayid.unique()
    rng = np.random.default_rng(0)
    fold = dict(zip(days, rng.integers(0, 5, len(days))))
    m["fold"] = m.dayid.map(fold)
    pr = np.zeros(len(m))
    coefs = []
    for k in range(5):
        tr, te = m.fold != k, m.fold == k
        A = m.loc[tr, [col + "_c", "pl1_c"]].values
        b = m.loc[tr, "sub_temp_c"].values
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        coefs.append(w)
        pr[te.values] = m.loc[te, [col + "_c", "pl1_c"]].values @ w
    base_in = rmse(m[col + "_c"] - m.sub_temp_c)
    new_in = rmse(pr - m.sub_temp_c)
    cw = np.mean(coefs, 0)
    out.append(f"   지연혼합(하루안, 교차적합): 계수 a {cw[0]:.3f} b {cw[1]:.3f} | 하루안 RMSE {base_in:.4f} → {new_in:.4f} "
               f"({100*(new_in/base_in-1):+.1f}%)")
    return out


cold_days = lab.loc[lab.pred_EXT10.notna(), "dayid"].unique()
p(f"EXT10 추운 날 수: {len(cold_days)}")
for line in analyse(lab, "pred_EXT10", "EXT10 추운날"):
    p(line)
for line in analyse(lab[lab.dayid.isin(cold_days)], "pred_DIAG10", "DIAG10 같은 추운날"):
    p(line)
for line in analyse(lab[~lab.dayid.isin(cold_days)], "pred_DIAG10", "DIAG10 나머지날"):
    p(line)
for line in analyse(lab, "pred_EXT12", "EXT12 추운날"):
    p(line)
c12 = lab.loc[lab.pred_EXT12.notna(), "dayid"].unique()
for line in analyse(lab[lab.dayid.isin(c12)], "pred_DIAG10", "DIAG10 EXT12날"):
    p(line)
# 온실별 (EXT10)
for fm in ("F13", "F47"):
    for line in analyse(lab[lab.farm == fm], "pred_EXT10", f"EXT10 {fm}"):
        p(line)
_f.close()
