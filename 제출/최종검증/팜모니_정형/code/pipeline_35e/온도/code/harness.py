# -*- coding: utf-8 -*-
"""Shared evaluation harness: build once, score many candidates the same way.

Validation policy (do not change without reading Claude/HANDOFF.md section 2)
---------------------------------------------------------------------------
The test rows are separated from the nearest label by >= 25 hours, median 69,
mean 3.6 DAYS.  Any CV whose held-out days sit further from labels than that
systematically punishes near-neighbour-in-time models, and any CV that sits
closer rewards them.  Rankings really do flip between the two.  So every
candidate here is scored on folds that copy the real test layout
E5 . T8 . E10 . T8 . E10 . T3 . E5, shifted into the dense part of the record.

`SHIFTS_A` is the set used for the submitted models (documented in HANDOFF).
`SHIFTS_B` is a disjoint confirmation set: a candidate that wins on A but not
on B was tuned to the placement, not to the problem.  Report both.

Target labels never enter feature construction; feature columns come from the
panel that features_v2 builds out of inputs only.
"""
import numpy as np
import pandas as pd

import env  # noqa: F401  (bootstraps DLL path / sys.path / DATA)
import common
import features_v2 as F2
from common import split_mask, rmse, TARGET_FARMS, USABLE

SEEDS = (7, 101, 2024)
LAYOUT = list(range(0, 5)) + list(range(15, 25)) + list(range(35, 45)) + list(range(50, 55))
SHIFTS_A = [70, 85, 100, 113, 126]
SHIFTS_B = [63, 77, 92, 107, 120]
DOMAIN_MARKS = ("rad_eff", "transp_pm", "root_dh", "heat_input", "rtr_",
                "transp_per_rad", "_cum", "screen_ins", "heat_screen")

_CACHE = {}


def load():
    """panel + the two labelled frames.  Cached per process."""
    if "panel" not in _CACHE:
        tX, ty, sX = common.load_raw()
        panel = F2.build(tX, sX)
        panel = panel.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
        panel["is_test"] = panel.row_id.isin(set(sX.row_id))
        panel["midnight"] = (panel.hour == 0).astype(float)
        _CACHE["panel"] = panel
        _CACHE["lab_t"] = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
        _CACHE["lab_e"] = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)
        _CACHE["test"] = panel[panel.is_test].reset_index(drop=True)
    return _CACHE["panel"], _CACHE["lab_t"], _CACHE["lab_e"]


def folds(kind="A"):
    shifts = {"A": SHIFTS_A, "B": SHIFTS_B}[kind]
    return [{f: {s + o for o in LAYOUT} for f in TARGET_FARMS} for s in shifts]


def gap_stats(lab, fds):
    g = []
    for fd in fds:
        trm, _ = split_mask(lab, fd)
        for f, vd in fd.items():
            tr = np.array(sorted(lab[trm & (lab.farm == f).values].day.unique()))
            g += [np.abs(tr - d).min() for d in vd]
    g = np.array(g)
    return dict(mean=float(g.mean()), median=float(np.median(g)), max=int(g.max()))


def score(lab, target, cols, factory, kind="A", seeds=SEEDS, return_oof=False):
    """Out-of-fold RMSE over the whole fold set.

    factory(seed) -> an unfitted estimator exposing fit/predict.
    Predictions are averaged over `seeds` inside each fold.
    """
    fds = folds(kind)
    oof = np.full(len(lab), np.nan)
    per = []
    for fd in fds:
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        ps = []
        for s in seeds:
            est = factory(s)
            est.fit(tr[cols], tr[target].values)
            ps.append(np.asarray(est.predict(va[cols]), float))
        p = np.mean(ps, axis=0)
        oof[np.where(vam)[0]] = p
        per.append(rmse(p, va[target].values))
    got = ~np.isnan(oof)
    r = rmse(oof[got], lab[target].values[got])
    out = (r, float(np.std(per)), per)
    return (out, oof) if return_oof else out


def views(panel):
    """The three column sets the submitted models used."""
    v_temp = F2.view(panel, "sub_temp")
    v_ec_full = F2.view(panel, "sub_ec")
    v_ec = [c for c in v_ec_full if not any(m in c for m in DOMAIN_MARKS)]
    v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]
    return dict(temp=v_temp, ec=v_ec, ec_full=v_ec_full, v5=v5, raw=list(USABLE))


if __name__ == "__main__":
    panel, lab_t, lab_e = load()
    v = views(panel)
    print("panel %s | labelled temp %d | labelled ec %d | test %d"
          % (panel.shape, len(lab_t), len(lab_e), int(panel.is_test.sum())))
    print("feature views: temp %d | ec %d | ec_full %d | v5 %d | raw %d"
          % tuple(len(v[k]) for k in ["temp", "ec", "ec_full", "v5", "raw"]))
    for f in TARGET_FARMS:
        d = sorted(lab_e[lab_e.farm == f].day.unique())
        print("  %s labelled EC days: %d..%d (n=%d)" % (f, d[0], d[-1], len(d)))
    for k in ("A", "B"):
        fd = folds(k)
        held = int(split_mask(lab_e, fd[0])[1].sum())
        print("folds %s: %s  gap=%s  held-out rows/fold=%d"
              % (k, {"A": SHIFTS_A, "B": SHIFTS_B}[k], gap_stats(lab_e, fd), held))
    print("real test gap for reference: mean 3.6 / median 3 / max 6")


# --------------------------------------------------------------------------
# Composition helpers (appended): equal/weighted blends of several estimators
# behind the same fit/predict interface `score` expects.
# --------------------------------------------------------------------------
class Blend:
    """Average several estimators' predictions.  factories: list of seed->est."""

    def __init__(self, factories, seed, weights=None):
        self.ests = [f(seed) for f in factories]
        n = len(self.ests)
        self.w = np.asarray(weights if weights is not None else [1.0 / n] * n, float)
        self.w = self.w / self.w.sum()

    def fit(self, X, y):
        for e in self.ests:
            e.fit(X, y)
        return self

    def predict(self, X):
        return np.sum([w * np.asarray(e.predict(X), float)
                       for w, e in zip(self.w, self.ests)], axis=0)


def blend_factory(factories, weights=None):
    return lambda s: Blend(factories, s, weights)
