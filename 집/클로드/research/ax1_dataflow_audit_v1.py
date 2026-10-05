# -*- coding: utf-8 -*-
"""AX1 (critic-driven data-flow audit, 2026-10-06 집 클로드).  Diagnosis only: test_X inputs are compared,
never used to fit or tune anything.  Segments: P1 = training pass 1 (day < 179), P2T = training pass 2
(labelled, day >= 179), TEST = evaluation records (pass 2).
A  raw input integrity per farm x segment: missing share, stuck runs (>= 6 h identical), ranges.
B  distribution shift P2T vs TEST: per raw input standardized mean difference (SMD) and KS; adversarial
   AUC (logistic on day-level input summaries, 5-fold CV) P2T vs TEST, and P1 vs P2T for reference.
C  model features (FULL + season + DP1, as submission_13 builds them): missing share and SMD, P2T vs TEST;
   season index distribution of TEST vs P2T.
D  where the pass-2 error sits (DIAG10 R3S OOF, pass-2 labelled rows): RMSE by hour; by sealed / open;
   share of SSE by hour block; day-level vs within-day split on pass-2 normal days.
E  label quirks on pass-2: flat runs (>= 6 h identical EC), largest hour-to-hour jumps, EC resolution."""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
RAW = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
       "act_heating", "act_circfan", "act_co2", "act_fog"]
tr = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + RAW); tr["src"] = "train"
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + RAW); te["src"] = "test"
X = pd.concat([tr, te]); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X["seg"] = np.where(X.src == "test", "TEST", np.where(X.day < 179, "P1", "P2T"))
print("A. raw input integrity (missing % | stuck>=6h share of hours %)")
X = X.sort_values(["farm", "day", "hour"])
for v in RAW:
    run = X.groupby(["farm", "day"])[v].transform(lambda s: s.groupby((s != s.shift()).cumsum()).transform("size"))
    X["stuck_" + v] = (run >= 6) & X[v].notna()
A = X.groupby(["farm", "seg"]).agg(**{"miss_" + v: (v, lambda s: 100 * s.isna().mean()) for v in ["in_temp", "in_hum", "in_co2"]},
                                   **{"stuck_" + v: ("stuck_" + v, lambda s: 100 * s.mean()) for v in ["in_temp", "in_hum", "in_co2", "out_temp"]})
print(A.round(1).to_string())
print("\nB. distribution shift P2T vs TEST (SMD = mean diff / pooled SD; KS p) per farm")
for f in ("F13", "F47"):
    a, b = X[(X.farm == f) & (X.seg == "P2T")], X[(X.farm == f) & (X.seg == "TEST")]
    rows = []
    for v in RAW:
        x1, x2 = a[v].dropna(), b[v].dropna()
        sd = np.sqrt((x1.var() + x2.var()) / 2) or 1
        rows.append((v, (x2.mean() - x1.mean()) / sd, ks_2samp(x1, x2).pvalue))
    T = pd.DataFrame(rows, columns=["var", "SMD", "KS_p"]).sort_values("SMD", key=abs, ascending=False)
    print("  %s top shifts:" % f, "; ".join("%s %+.2f (p %.3f)" % (r.var, r.SMD, r.KS_p) for r in T.head(5).itertuples()))


def day_summ(S):
    g = S.groupby(["farm", "day"])
    out = g[RAW].mean().add_suffix("_m").join(g[["in_temp", "in_hum", "in_co2", "out_temp", "out_rad"]].std().add_suffix("_s"))
    out["seg"] = g.seg.first()
    return out.reset_index()


DS = day_summ(X)
for f in ("F13", "F47"):
    for s1, s2 in (("P2T", "TEST"), ("P1", "P2T")):
        Z = DS[(DS.farm == f) & DS.seg.isin([s1, s2])]
        y = (Z.seg == s2).astype(int).values; F = [c for c in Z.columns if c.endswith(("_m", "_s"))]
        m = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=.3, max_iter=3000))
        k = min(5, int(y.sum()), int((1 - y).sum()))
        pr = cross_val_predict(m, Z[F].values, y, cv=StratifiedKFold(k, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        print("  adversarial AUC %s %s vs %s (days %d/%d): %.2f" % (f, s1, s2, (1 - y).sum(), y.sum(), roc_auc_score(y, pr)))
# C. model features
spec = importlib.util.spec_from_file_location("m13", r"C:\work\farmai\집\클로드\local\submission14_ec_sg2_20261005\pkg\season.py")
print("\nC. model features (FULL + DP1) P2T vs TEST")
sys.path.insert(0, r"C:\work\farmai\집\클로드\local\submission14_ec_sg2_20261005\pkg")
os.environ.setdefault("EC_CORE_DEPENDENCY_PATH", "")
src = open(r"C:\work\farmai\집\클로드\local\submission14_ec_sg2_20261005\pkg\model.py", encoding="utf-8").read()
ns = {"__file__": os.path.join(r"C:\work\farmai\집\클로드\local\submission14_ec_sg2_20261005\pkg", "model.py"), "__name__": "lib"}
exec(src.split("def shrink")[0].replace("import env\nimport env_extra", "").replace("import sklearn,lightgbm,torch,tabpfn", "").replace("from tabpfn import TabPFNRegressor", "").replace("from tabpfn.constants import ModelVersion", "").replace("import sg2post", ""), ns)
Fe = ns["features"](X[["row_id"] + ns["RAW"]].reset_index(drop=True))
Fe["seg"] = X.set_index("row_id").loc[Fe.row_id, "seg"].values; Fe["farm"] = Fe.row_id.str[:3]
cols = [c for c in ns["DAY_FULL"] + ns["OPS"] if c not in ("day",)]
rows = []
for f in ("F13", "F47"):
    a, b = Fe[(Fe.farm == f) & (Fe.seg == "P2T")], Fe[(Fe.farm == f) & (Fe.seg == "TEST")]
    for c in cols:
        x1, x2 = a[c].astype(float), b[c].astype(float)
        sd = np.sqrt((x1.var() + x2.var()) / 2) or 1
        rows.append((f, c, 100 * x1.isna().mean(), 100 * x2.isna().mean(), (x2.mean() - x1.mean()) / sd))
T = pd.DataFrame(rows, columns=["farm", "feat", "miss_P2T%", "miss_TEST%", "SMD"])
print("  features with missing share differing by > 1 point:", T[(T["miss_P2T%"] - T["miss_TEST%"]).abs() > 1][["farm", "feat", "miss_P2T%", "miss_TEST%"]].round(1).to_dict("records"))
print("  largest |SMD| features:"); print(T.reindex(T.SMD.abs().sort_values(ascending=False).index).head(10).round(2).to_string(index=False))
ev = pd.read_csv(r"C:\work\farmai\집\클로드\local\submission14_ec_sg2_20261005\run1\evaluation_day_to_season.csv")
trs = pd.read_csv(r"C:\work\farmai\집\클로드\local\submission14_ec_sg2_20261005\run1\training_day_to_season.csv")
for f in ("F13", "F47"):
    print("  season %s: TEST %.1f..%.1f (median %.1f) | P2T %.1f..%.1f (median %.1f)" % (f, ev[ev.farm == f].season.min(), ev[ev.farm == f].season.max(),
          ev[ev.farm == f].season.median(), trs[(trs.farm == f) & (trs.day >= 179)].season.min(), trs[(trs.farm == f) & (trs.day >= 179)].season.max(),
          trs[(trs.farm == f) & (trs.day >= 179)].season.median()))
# D. error location
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[(O.validator == "DIAG10") & (O.day >= 179)].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1); O["e"] = O.p - O.sub_ec
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
N = O[O.dm < 1].copy()
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
print("\nD. pass-2 NORMAL days (%d rows): RMSE by hour block / bias" % len(N))
N["blk"] = pd.cut(N.hour, [-1, 5, 11, 17, 23], labels=["0-5", "6-11", "12-17", "18-23"])
print(N.groupby("blk", observed=True).e.agg([r, "mean"]).round(3).T.to_string())
N["pmd"] = N.groupby(["farm", "day"]).p.transform("mean"); N["ed"] = N.pmd - N.dm
print("  day-level share of SSE %.0f%%; day-level bias mean %+.3f; by farm:" % (100 * (N.ed ** 2).sum() / (N.e ** 2).sum(), N.groupby(["farm", "day"]).ed.first().mean()),
      N.groupby("farm").apply(lambda g: "%.3f (bias %+.3f)" % (r(g.e), g.e.mean())).to_dict())
# E. label quirks pass-2
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
Y = Y.sort_values(["farm", "day", "hour"]); Y["d"] = Y.groupby(["farm", "day"]).sub_ec.diff()
run = Y.groupby(["farm", "day"]).sub_ec.transform(lambda s: s.groupby((s != s.shift()).cumsum()).transform("size"))
for nm, m in (("P1", Y.day < 179), ("P2T", Y.day >= 179)):
    g = Y[m]
    print("E. %s labels: flat>=6h rows %.1f%%, |hourly jump|>.2 rows %.2f%%, decimals .001 grid %.0f%%" % (
        nm, 100 * (run[m] >= 6).mean(), 100 * (g.d.abs() > .2).mean(), 100 * np.isclose(g.sub_ec * 1000, np.round(g.sub_ec * 1000)).mean()))
