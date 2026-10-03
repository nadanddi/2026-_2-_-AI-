# -*- coding: utf-8 -*-
"""PR1 (diagnostic; 2026-10-04 집 클로드).  User question: can the model's confidence be
expressed as a probability, and do under-predicted high-EC days show it?
For each DIAG10 fold: ExtraTrees (core.et(7), FS = FULL - day + season, DC4 season)
as in the R3S ET member; per-tree predictions for validation rows -> per-tree DAY
means -> distribution over 600 trees per day.  Report per day: mean, q10/q50/q90,
P(day >= 1.0), P(day >= 1.5).  (Leaf-1 ExtraTrees leaves hold training labels, so the
tree spread approximates the conditional spread, like a quantile forest.)
Checks: calibration of P(>=1) and P(>=1.5) over all days (bins); on the 31 high days,
whether the under-predicted ones carried a larger upper tail (q90, P(>=1.5)); PIT
coverage of the q10-q90 interval."""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
OUT = os.path.join(env.LOCAL, "pr1_day_dist.csv")
if not os.path.exists(OUT):
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    rows = []
    for name, i, vd in fds:
        if name != "DIAG10": continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        m = core.et(7); m.fit(tr[FS], tr.sub_ec.values)
        Xv = m.steps[0][1].transform(va[FS]); et = m.steps[-1][1]
        T = np.stack([t.predict(Xv) for t in et.estimators_], axis=1)        # rows x trees
        key = list(zip(va.farm, va.day))
        dfT = pd.DataFrame(T); dfT["farm"], dfT["day"] = va.farm.values, va.day.values
        dm = dfT.groupby(["farm", "day"]).mean()                              # day x trees
        y = va.groupby(["farm", "day"]).sub_ec.mean()
        for (f, d), v in dm.iterrows():
            v = v.values
            rows.append(dict(farm=f, day=d, fold=i, y=y[(f, d)], mean=v.mean(), q10=np.quantile(v, .1), q50=np.quantile(v, .5),
                             q90=np.quantile(v, .9), p_ge1=(v >= 1).mean(), p_ge15=(v >= 1.5).mean()))
        print("fold %d done" % i, flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
D = pd.read_csv(OUT)
D["hi"] = D.y >= 1; D["vhi"] = D.y >= 1.5; D["err"] = D["mean"] - D.y
print("\nCALIBRATION over %d days" % len(D))
for col, tgt in (("p_ge1", "hi"), ("p_ge15", "vhi")):
    b = pd.cut(D[col], [-.01, .05, .2, .4, .6, .8, 1.0])
    print(" %s:" % col); print(D.groupby(b, observed=True).agg(days=(tgt, "size"), mean_prob=(col, "mean"), actual_rate=(tgt, "mean")).round(2).to_string())
cov = ((D.y >= D.q10) & (D.y <= D.q90)).mean()
print("q10-q90 interval coverage (nominal ~80%%): %.2f | high days: %.2f" % (cov, ((D.y >= D.q10) & (D.y <= D.q90))[D.hi].mean()))
H = D[D.hi].copy()
H["grp"] = pd.cut(H.err, [-9, -.5, -.2, .2, 9], labels=["under >.5", "under .2-.5", "within .2", "over"])
print("\nHIGH-EC days by error group")
print(H.groupby("grp", observed=True).agg(n=("y", "size"), true=("y", "mean"), mean=("mean", "mean"), q10=("q10", "mean"), q90=("q90", "mean"),
      p_ge1=("p_ge1", "mean"), p_ge15=("p_ge15", "mean")).round(2).to_string())
print(H.sort_values("err")[["farm", "day", "y", "mean", "q10", "q90", "p_ge1", "p_ge15"]].round(2).to_string(index=False))
