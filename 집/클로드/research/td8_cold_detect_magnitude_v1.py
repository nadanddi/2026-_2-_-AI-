# -*- coding: utf-8 -*-
"""TD8: can the given inputs tell (1) WHETHER the substrate will be cold and (2) HOW cold?  (2026-10-10 집 클로드,
user: "추운 날을 주어진 데이터로 그 날의 배지 온도가 추울지와 얼마나 추울지를 맞출 수 있어?")  Diagnostic, F13/F47 only
(other farms: 6b.24 cold sub-in totally different, 6b.29 rejected).  Fixed before running:
 (1) DETECTION  target = row sub_temp < 8.  Score = causal inputs only (in_temp now, ewm3 of in_temp).  Report AUC,
     and day-level: day has any sub<8 hour vs (a) day min ewm3 (all hours' inputs of that day = available hour by hour),
     (b) EARLY: ewm3 at hour 0 only.  'detectable' iff AUC >= .95.
 (2) MAGNITUDE on EXT10 rows (cold-ish holdout days, W40G-S OOF from TT1 checkpoints = 9th-round members):
     - error by in_temp bin (<4, 4-6, 6-8, 8-10, 10-12, >=12): RMSE, bias, n; plus count of train/test rows per bin.
     - day level vs within-day split; oracle 'know the daily offset' RMSE.
     - is the DAILY offset predictable?  day features (in mean/min, out mean, heating, fan, thermal, vent, hum, co2,
       rad, in-out) -> leave-one-day-out ridge (alpha 10, standardized) R^2 with day bootstrap 95% CI.
       'signal' iff R^2 > .10 and CI lower > 0 in both farms pooled AND same-sign Spearman in F13 and F47 for the top
       feature.  Otherwise label '정보 부족'.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td8_cold_detect_magnitude_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import common
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
CK = os.path.join(env.LOCAL, "tt1_ckpt")
FEATS = ["in_mean", "in_min", "out_mean", "heat", "fan", "thermal", "vent", "hum", "co2", "rad", "in_out"]


def prep(a):
    a = a[a.farm.isin(["F13", "F47"])].sort_values(["farm", "t"]).copy()
    a["ewm3"] = a.groupby("farm").in_temp.transform(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
    return a


def dayfeat(a):
    d = a.groupby(["farm", "day"]).agg(in_mean=("in_temp", "mean"), in_min=("in_temp", "min"), out_mean=("out_temp", "mean"),
                                       heat=("act_heating", "mean"), fan=("act_circfan", "mean"), thermal=("act_thermal", "mean"),
                                       vent=("act_vent", "mean"), hum=("in_hum", "mean"), co2=("in_co2", "mean"), rad=("out_rad", "sum"))
    d["in_out"] = d.in_mean - d.out_mean
    return d


def main():
    tX, ty, sX = common.load_raw()
    Tr = prep(tX.merge(ty[["row_id", "sub_temp"]], on="row_id")); Te = prep(sX)
    print("== (1) 감지: 배지 < 8℃ 행 %d / %d" % ((Tr.sub_temp < 8).sum(), len(Tr)))
    y = (Tr.sub_temp < 8).astype(int)
    print("  행 AUC  in_temp %.3f | ewm3 %.3f" % (roc_auc_score(y, -Tr.in_temp.fillna(99)), roc_auc_score(y, -Tr.ewm3)))
    D = Tr.groupby(["farm", "day"]).agg(cold=("sub_temp", lambda s: (s < 8).any()), mn=("ewm3", "min"))
    D["h0"] = Tr[Tr.hour == 0].set_index(["farm", "day"]).ewm3
    print("  날 AUC  (배지<8 시각이 있는 날 %d일) 하루 최저 ewm3 %.3f | 0시 ewm3만 %.3f" % (D.cold.sum(), roc_auc_score(D.cold, -D.mn), roc_auc_score(D.cold, -D.h0.fillna(99))))
    for thr in (8, 9, 10):
        print("    규칙 '하루 최저 ewm3 < %d': 맞힘 %d/%d, 헛경보 %d" % (thr, ((D.mn < thr) & D.cold).sum(), D.cold.sum(), ((D.mn < thr) & ~D.cold).sum()))

    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    G = G[G.validator == "EXT10"].copy()
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    G["p"] = np.mean([.4 * G["base_REF_%d" % s] + (.2 + .4 * (1 - g)) * G.codex_REF + .4 * g * G["pfn_%s" % f] for s in (7, 101) for f in ("A", "B")], 0)
    G["e"] = G.p - G.sub_temp
    print("\n== (2) 크기: EXT10 (추운 쪽 제외 검증) %d일, 9회차 W40G-S RMSE %.3f, 치우침 %+.3f" % (G.groupby(["farm", "day"]).ngroups, r(G.e), G.e.mean()))
    bins = [-np.inf, 4, 6, 8, 10, 12, np.inf]; lab = ["<4", "4-6", "6-8", "8-10", "10-12", ">=12"]
    G["b"] = pd.cut(G.in_temp, bins, labels=lab, right=False)
    trb = pd.cut(Tr.in_temp, bins, labels=lab, right=False).value_counts(); teb = pd.cut(Te.in_temp, bins, labels=lab, right=False).value_counts()
    print("  실내온도 구간 | 학습 행 | 평가 행 | EXT10 행 RMSE 치우침 | 배지-실내 평균(학습)")
    for b in lab:
        x = G[G.b == b]; t = Tr[pd.cut(Tr.in_temp, bins, labels=lab, right=False) == b]
        print("   %-6s | %5d | %4d | %4d %.3f %+.3f | %+.2f" % (b, trb.get(b, 0), teb.get(b, 0), len(x), r(x.e) if len(x) else np.nan, x.e.mean() if len(x) else np.nan, (t.sub_temp - t.in_temp).mean() if len(t) else np.nan))
    k = G.groupby(["farm", "day"]).e.transform("mean")
    print("  하루 수준 몫 %.0f%% | 하루 오프셋을 안다면(오라클) RMSE %.3f → %.3f" % (100 * (k ** 2).sum() / (G.e ** 2).sum(), r(G.e), r(G.e - k)))
    off = G.groupby(["farm", "day"]).e.mean().rename("off")
    F = dayfeat(Tr).join(off, how="inner").dropna()
    X = ((F[FEATS] - F[FEATS].mean()) / F[FEATS].std()).values; yv = F.off.values; pr = np.zeros(len(F))
    for i in range(len(F)):
        m = np.arange(len(F)) != i
        pr[i] = Ridge(alpha=10).fit(X[m], yv[m]).predict(X[i:i + 1])[0]
    R2 = 1 - ((yv - pr) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(3000):
        j = rng.integers(0, len(F), len(F)); bs.append(1 - ((yv[j] - pr[j]) ** 2).sum() / ((yv[j] - yv[j].mean()) ** 2).sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("  하루 오프셋 예측(날 하나 빼기 Ridge, %d일): R² %+.3f  95%% CI [%+.3f, %+.3f]" % (len(F), R2, lo, hi))
    print("  특징별 Spearman (F13 / F47):")
    best = None
    for c in FEATS:
        a = spearmanr(F.loc["F13", c], F.loc["F13", "off"])[0]; b = spearmanr(F.loc["F47", c], F.loc["F47", "off"])[0]
        print("    %-8s %+.2f / %+.2f" % (c, a, b))
        if best is None or abs(a + b) > abs(best[1] + best[2]):
            best = (c, a, b)
    sig = R2 > .10 and lo > 0 and np.sign(best[1]) == np.sign(best[2])
    print("  판정: %s (최상위 특징 %s)" % ("신호 있음" if sig else "정보 부족", best[0]))


if __name__ == "__main__":
    main()
