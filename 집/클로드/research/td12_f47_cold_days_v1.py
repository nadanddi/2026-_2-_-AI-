# -*- coding: utf-8 -*-
"""TD12: F47 cold days one by one (2026-10-10 집 클로드, user: "F47 추운 날 날짜별로 직접 확인해봐. 이웃 방식이 안 통할 수도").
Descriptive (data-generation forensics).  Residual = W40G-S DIAG10 OOF (seed 7, PFN A) - label.
Cold day = F47 day whose in_temp (raw) < 8 at any hour.  Fingerprints fixed before looking:
 H1 dong: offset mean by diagnostic dong (st_dong_assign, label-derived diagnostic only) differs by >= .4 C.
 H2 persistence (neighbour methods work): corr(offset_d, offset_{d-1}) over all F47 days >= .3, and among cold days with
    a cold/any neighbour the sign agrees in >= 70% -> otherwise neighbour correction will not work.
 H3 operations: |Spearman(offset, x)| >= .4 for heating/curtain/fan/vent/out_temp/in_min within the 24 cold days
    (n small - descriptive).
 H4 shape: print the worst 4 days hourly (in, sub, pred, heat, thermal, fan, out) next to the previous F47 day.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td12_f47_cold_days_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import common
from scipy.stats import spearmanr
CK = os.path.join(env.LOCAL, "tt1_ckpt")


def main():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    G = G[(G.validator == "DIAG10") & (G.farm == "F47")].copy()
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    G["p"] = .4 * G.base_REF_7 + (.2 + .4 * (1 - g)) * G.codex_REF + .4 * g * G.pfn_A
    tX, ty, _ = common.load_raw()
    X = tX[tX.farm == "F47"].merge(ty[["row_id", "sub_temp"]], on="row_id")
    X = X.merge(G[["row_id", "p"]], on="row_id", how="left").sort_values("t")
    X["e"] = X.p - X.sub_temp
    A = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).query("farm=='F47'").set_index("day").dong
    D = X.groupby("day").agg(off=("e", "mean"), rmse=("e", lambda s: np.sqrt((s ** 2).mean())), in_min=("in_temp", "min"), in_mean=("in_temp", "mean"),
                             sub_mean=("sub_temp", "mean"), out=("out_temp", "mean"), heat=("act_heating", "mean"), thermal=("act_thermal", "mean"),
                             fan=("act_circfan", "mean"), vent=("act_vent", "mean"), hum=("in_hum", "mean"))
    D["gap"] = D.sub_mean - D.in_mean
    D["dong"] = A.reindex(D.index)
    D["jump0"] = [X[(X.day == d) & (X.hour == 0)].sub_temp.mean() - X[(X.day == d - 1) & (X.hour == 23)].sub_temp.mean() if (d - 1) in D.index else np.nan for d in D.index]
    D["prev_off"] = D.off.shift(1).where(pd.Series(D.index, index=D.index).diff() == 1)
    D["next_off"] = D.off.shift(-1).where(pd.Series(D.index, index=D.index).diff(-1) == -1)
    C = D[D.in_min < 8]
    pd.set_option("display.width", 250)
    print("== F47 추운 날 %d일 (날짜순). off=하루 평균 오차(예측-정답, +면 높게 예측), gap=배지-실내 평균, jump0=전날23시→0시 배지 변화" % len(C))
    print(C[["dong", "off", "rmse", "prev_off", "next_off", "jump0", "in_min", "in_mean", "sub_mean", "gap", "out", "heat", "thermal", "fan", "vent", "hum"]].round(2).to_string())
    print("\n== H1 동별 하루 오차 평균: 추운 날", C.groupby("dong").off.agg(["mean", "count"]).round(2).to_dict(), "| 전체 F47", D.groupby("dong").off.agg(["mean", "count"]).round(2).to_dict())
    v = D[["off", "prev_off"]].dropna()
    print("== H2 이웃: 모든 F47 날 corr(오늘, 어제) %.2f (n%d)" % (np.corrcoef(v.off, v.prev_off)[0, 1], len(v)))
    vc = C[["off", "prev_off", "next_off"]]
    for nb in ("prev_off", "next_off"):
        z = vc[["off", nb]].dropna()
        print("   추운 날 vs %s: 부호 일치 %d/%d, corr %.2f" % (nb, (np.sign(z.off) == np.sign(z[nb])).sum(), len(z), np.corrcoef(z.off, z[nb])[0, 1] if len(z) > 2 else np.nan))
    sd = A.reindex(D.index)
    same = []
    for d in C.index:
        prev = [k for k in range(d - 1, d - 8, -1) if k in D.index and sd.get(k) == sd.get(d)]
        if prev:
            same.append((C.off[d], D.off[prev[0]]))
    s = np.array(same)
    print("   추운 날 vs 같은 동의 직전 기록: 부호 일치 %d/%d, corr %.2f" % ((np.sign(s[:, 0]) == np.sign(s[:, 1])).sum(), len(s), np.corrcoef(s.T)[0, 1]))
    print("== H3 추운 날 24일 안에서 Spearman(하루 오차, x):", {c: round(spearmanr(C[c], C.off)[0], 2) for c in ("in_min", "in_mean", "out", "heat", "thermal", "fan", "vent", "hum", "jump0", "gap")})
    print("\n== H4 최악 4일 시간별 (앞: 전날 같은 시각 배지)")
    for d in C.rmse.nlargest(4).index:
        x = X[X.day == d].set_index("hour"); pv = X[X.day == d - 1].set_index("hour").sub_temp
        print("\n-- day %d (동 %s, off %+.2f, rmse %.2f)" % (d, C.dong[d], C.off[d], C.rmse[d]))
        T = pd.DataFrame({"실내": x.in_temp, "배지": x.sub_temp, "예측": x.p, "오차": x.e, "전날배지": pv.reindex(x.index), "외기": x.out_temp, "난방": x.act_heating, "커튼": x.act_thermal, "팬": x.act_circfan, "환기": x.act_vent})
        print(T.round(1).T.to_string())


if __name__ == "__main__":
    main()
