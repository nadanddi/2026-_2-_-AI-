# -*- coding: utf-8 -*-
"""TD27: can F32's nutrient-pump record (act_pump = 양액 펌프 가동비율, 문제설명서 2쪽; only F32 has it) be transferred to
F13/F47 as an estimated hidden irrigation state?  (2026-10-10 집 클로드, user: "13·47은 없고 32만 있으면 이걸 써서 13·47의 무언가를
예측하라는 거 아닐까")  Allowed form: a model trained on public training data (F32) applied to F13/F47 inputs.  Fixed before running:
 (1) F32: classifier pump>0 from the 9 shared inputs (out_temp, out_rad, out_wspd, in_temp, in_hum, in_co2, act_vent,
     act_circfan, act_co2) + hour sin/cos + their 3 h/6 h causal means; day-block 5-fold AUC.
 (2) F32: does the pump move the substrate beyond air?  OLS of hourly d(sub) on d(in_temp), (in - sub)_{t-1}, hour terms,
     pump_t, pump_{t-1}; report pump coefficients (C/h) with day-bootstrap CI.
 (3) Transfer: fit (1) on all F32 rows, predict pump probability for F13/F47 rows (inputs only; F13/F47 in_hum scale differs
     - F32 mean 26 vs 70 - so a variant without in_hum is also reported); daily mean estimate vs W40G-S DIAG10 daily
     residual OFF and afternoon deficit DEF, partial Spearman on day in_temp; rough days excluded.
     'worth a model test' iff same sign in F13 and F47, min |rho| >= .2, both p < .01.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td27_f32_pump_transfer_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import common
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
SH = ["out_temp", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_circfan", "act_co2"]


def feats(x, cols):
    x = x.sort_values("t").copy()
    for c in cols:
        for h in (3, 6):
            x[c + "_m%d" % h] = x.groupby("farm")[c].transform(lambda s: s.rolling(h, min_periods=1).mean())
    x["hs"], x["hc"] = np.sin(2 * np.pi * x.hour / 24), np.cos(2 * np.pi * x.hour / 24)
    return x, cols + [c + "_m%d" % h for c in cols for h in (3, 6)] + ["hs", "hc"]


def main():
    tX, ty, _ = common.load_raw()
    a = tX.merge(ty[["row_id", "sub_temp"]], on="row_id")
    F32 = a[a.farm == "F32"].copy()
    print("== (1) F32 양액 펌프 가동(>0)을 공통 입력으로 예측")
    for nm, cols in (("공통 9열", SH), ("습도 제외", [c for c in SH if c != "in_hum"])):
        x, X = feats(F32, cols); x = x.dropna(subset=["act_pump"]); y = (x.act_pump > 0).astype(int).values
        days = np.sort(x.day.unique()); p = np.zeros(len(x))
        for k in range(5):
            te = np.isin(x.day, days[k::5])
            m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=.05, verbose=-1).fit(x.loc[~te, X], y[~te])
            p[te] = m.predict_proba(x.loc[te, X])[:, 1]
        print("  %s: AUC %.3f (펌프 가동 비율 %.2f)" % (nm, roc_auc_score(y, p), y.mean()))
    print("\n== (2) F32: 펌프가 공기와 별개로 배지를 움직이나")
    x = F32.sort_values("t").reset_index(drop=True); same = x.t.diff() == 1
    z = pd.DataFrame({"ds": x.sub_temp.diff(), "di": x.in_temp.diff(), "gap": (x.in_temp - x.sub_temp).shift(1),
                      "p0": (x.act_pump > 0).astype(float), "p1": (x.act_pump.shift(1) > 0).astype(float),
                      "hs": np.sin(2 * np.pi * x.hour / 24), "hc": np.cos(2 * np.pi * x.hour / 24), "day": x.day})[same].dropna()
    A = lambda d: np.c_[np.ones(len(d)), d.di, d.gap, d.hs, d.hc, d.p0, d.p1]
    b = np.linalg.lstsq(A(z), z.ds, rcond=None)[0]
    rng = np.random.default_rng(0); u = z.day.unique(); bs = []
    for _ in range(500):
        dd = rng.choice(u, len(u)); zz = pd.concat([z[z.day == d] for d in dd]); bs.append(np.linalg.lstsq(A(zz), zz.ds, rcond=None)[0][5:7])
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
    print("  펌프 지금 켜짐 계수 %+.3f℃/h [%+.3f, %+.3f], 한 시간 전 켜짐 %+.3f [%+.3f, %+.3f] (실내변화 %.2f, 따라감 %.3f)" % (b[5], lo[0], hi[0], b[6], lo[1], hi[1], b[1], b[2]))
    print("\n== (3) F32로 배운 펌프 추정값을 F13·F47에 적용 → 온도 오차와의 관계")
    CK = os.path.join(env.LOCAL, "tt1_ckpt"); G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in os.listdir(CK)]); G = G[G.validator == "DIAG10"]
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    G["e"] = np.mean([.4 * G["base_REF_%d" % s] + (.2 + .4 * (1 - g)) * G.codex_REF + .4 * g * G["pfn_%s" % f] for s in (7, 101) for f in "AB"], 0) - G.sub_temp
    T = a[a.farm.isin(["F13", "F47"])].merge(G[["row_id", "e"]], on="row_id")
    for nm, cols in (("공통 9열", SH), ("습도 제외", [c for c in SH if c != "in_hum"])):
        x32, X = feats(F32.dropna(subset=["act_pump"]), cols)
        m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=.05, verbose=-1).fit(x32[X], (x32.act_pump > 0).astype(int))
        xt, _ = feats(T, cols); xt["pp"] = m.predict_proba(xt[X])[:, 1]
        rows = []
        for (f, d), q in xt.groupby(["farm", "day"]):
            if (f, d) in RD:
                continue
            q = q.set_index("hour")
            rows.append(dict(farm=f, day=d, PP=q.pp.mean(), PPday=q.pp.loc[9:16].mean(), OFF=q.e.mean(),
                             DEF=q.e.loc[13:16].mean() - q.e.get(8, np.nan), tin=q.in_temp.mean()))
        D = pd.DataFrame(rows)
        print("  [%s] F13·F47 추정 펌프 가동 평균 F13 %.2f / F47 %.2f (F32 실제 %.2f)" % (nm, D[D.farm == "F13"].PP.mean(), D[D.farm == "F47"].PP.mean(), (F32.act_pump > 0).mean()))
        for t in ("OFF", "DEF"):
            for c in ("PP", "PPday"):
                res = []
                for f in ("F13", "F47"):
                    zz = D[D.farm == f][[c, t, "tin"]].dropna()
                    rx = zz[c] - np.poly1d(np.polyfit(zz.tin, zz[c], 1))(zz.tin); ry = zz[t] - np.poly1d(np.polyfit(zz.tin, zz[t], 1))(zz.tin)
                    res.append(spearmanr(rx, ry))
                ok = np.sign(res[0][0]) == np.sign(res[1][0]) and min(abs(res[0][0]), abs(res[1][0])) >= .2 and max(res[0][1], res[1][1]) < .01
                print("    %s ~ %-5s F13 %+.2f (p %.3f) / F47 %+.2f (p %.3f) %s" % (t, c, res[0][0], res[0][1], res[1][0], res[1][1], "← 모델 시험 가치" if ok else ""))


if __name__ == "__main__":
    main()
