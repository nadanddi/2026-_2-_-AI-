# -*- coding: utf-8 -*-
"""SF1 (2026-10-12 집 클로드; outdoor-expert finding: season table puts F47 pass-1 records on F47's own numbering although the same
F13 date differs by +3..-8 days (F47 61-150: about -4..-5)).  HYPOTHESIS AND RULE FIXED BEFORE RUNNING.
SF1 = COMMON CALENDAR: for each fold, F47 pass-1 TRAINING records with an exact same-date F13 pass-1 TRAINING record (raw outdoor
24 h vectors, >= 90% equal cells) give offset = F13 number - F47 number; offsets are smoothed by a rolling median (window 7, centred)
along F47 number and linearly interpolated to every F47 number < 179 (constant beyond the ends).  Common season of an F47 pass-1
record d = d + off(d); F13 pass-1 = d.  The pass-2 PAV anchors use the COMMON season of the matched twins.  Validation/evaluation rows
use the same rules (pass-1 F47 val: d + off(d); pass-2: PAV interp as before).  Only training weather is used -> rule-safe.
Arms: BASE (season.mapping as submitted) vs SF1; the model is retrained for each arm (training seasons differ).
Model / validators / seeds: as SE1 (proxy ET .6 + LGB .3, DIAG10 / BLK10 / HOLE5, seeds 7 / 101 / 2024).
PRE-FIXED PASS: SF1 RMSE < BASE on pass-2 val rows in all 9 cells AND on ALL val rows in all 9 cells AND DIAG10 record bootstrap
P(SF1 worse) < .025 for every seed (pass-2 records).
"""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import env_extra  # noqa: F401  (torch/tabpfn needed by model.py import)
import importlib.util, os, time
import numpy as np, pandas as pd
from pathlib import Path

PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
sys.path.insert(0, PK)
import model as M
from season import mapping, vectors, W
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor

OUT = Path(r"C:\work\farmai\집\클로드\research\sf1_season_common")
SEEDS = (7, 101, 2024)
THR, HMIN = .05, 5
tr = pd.read_pickle(r"C:\work\farmai\집\클로드\research\audit_ec\tr.pkl")
tr["day"] = tr.row_id.str[4:7].astype(int)
x = pd.read_csv(os.path.join(PK, "data", "train_X.csv"), usecols=["row_id"] + W)
VEC = vectors(x)
ALL = sorted(set(zip(tr.farm, tr.day.astype(int))))

P3 = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "ec_restart_phase3_20261001_v1")
sys.path.insert(0, P3)
spec = importlib.util.spec_from_file_location("p3_readonly", os.path.join(P3, "run_benchmark.py"))
p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)
_, _, _, LOCK, _, FDS = p3.prepare()


def common_mapping(tdays, qdays, vec):
    """season.mapping with F47 pass-1 records moved onto the F13 calendar (training weather only)."""
    from season import pav
    p1 = [(f, int(d)) for f, d in tdays.itertuples(index=False, name=None) if d < 179]
    f13 = [k for k in p1 if k[0] == "F13"]; f47 = sorted(k for k in p1 if k[0] == "F47")
    A = np.stack([vec[k].ravel() for k in f13]); offs = {}
    for k in f47:
        eq = (np.abs(A - vec[k].ravel()) < 1e-6).mean(1); j = int(np.argmax(eq))
        if eq[j] >= .9: offs[k[1]] = f13[j][1] - k[1]
    o = pd.Series(offs).sort_index(); o = o.rolling(7, center=True, min_periods=1).median()
    off = lambda d: float(np.interp(d, o.index.values, o.values))
    cs = lambda k: float(k[1]) if k[0] == "F13" else k[1] + off(k[1])
    reference = np.stack([vec[k] for k in p1])
    mu = np.nanmean(reference, axis=(0, 2)); sd = np.nanstd(reference, axis=(0, 2)); sd[sd == 0] = 1
    ref = (reference - mu[None, :, None]) / sd[None, :, None]
    seasons = {k: cs(k) for k in p1}; fits = {}
    for farm in ["F13", "F47"]:
        days = sorted(int(d) for f, d in tdays.itertuples(index=False, name=None) if f == farm and d >= 179)
        x, y, nearest = [], [], []
        for d in days:
            q = (vec[(farm, d)] - mu[:, None]) / sd[:, None]
            dist = np.sqrt(np.nanmean((ref - q) ** 2, axis=(1, 2)))
            nearest.append(cs(p1[int(np.nanargmin(dist))])); sel = np.flatnonzero(dist <= .05)
            if len(sel): x.append(d); y.append(float(np.mean([cs(p1[i]) for i in sel])))
        if len(x) < 2: x, y = days, nearest
        fitted = pav(y); x = np.asarray(x, float); fits[farm] = (x, fitted)
        for d in days: seasons[(farm, d)] = float(np.interp(d, x, fitted))
    out = []
    for farm, d in qdays.itertuples(index=False, name=None):
        if d < 179: out.append(float(d) if farm == "F13" else d + off(d))
        else: out.append(float(np.interp(d, *fits[farm])))
    return seasons, np.asarray(out), {"standardization": {"mu": mu.tolist(), "sd": sd.tolist()}, "n_offsets": len(offs)}


def folds():
    out = []
    for name, i, vd in FDS:
        if name == "DIAG10":
            out.append(("DIAG10", i, set(vd), set(LOCK)))
    fold = ((tr.day // 10) * 7 + (tr.farm == "F47") * 3) % 5
    for k in range(5):
        out.append(("BLK10", k, set(zip(tr.farm[fold == k], tr.day[fold == k].astype(int))), set()))
    ch = {}
    for f in ("F13", "F47"):
        days = sorted(d for ff, d in ALL if ff == f and d >= 179); runs = []; cur = [days[0]]
        for d in days[1:]:
            if d == cur[-1] + 1: cur.append(d)
            else: runs.append(cur); cur = [d]
        runs.append(cur); c = []
        for r in runs:
            for s in range(0, len(r), 5): c.append(r[s:s + 5])
        ch[f] = c
    assert len(ch["F13"]) == len(ch["F47"]) == 7, ch
    for j in range(7):
        out.append(("HOLE5", j, {("F13", d) for d in ch["F13"][j]} | {("F47", d) for d in ch["F47"][j]}, set()))
    return out


def twin_seasons(vrecs, refs, rseason, mu, sd, full=False):
    """vrecs: list of (farm, day) pass-2 validation records -> dict (f,d) -> array(24) season or nan (no match)."""
    R = np.stack([(VEC[k] - mu[:, None]) / sd[:, None] for k in refs]); rs = np.asarray(rseason, float); out = {}
    for k in vrecs:
        qz = (VEC[k] - mu[:, None]) / sd[:, None]; s = np.full(24, np.nan)
        if full:
            dist = np.sqrt(np.nanmean((R - qz[None]) ** 2, axis=(1, 2))); sel = dist <= THR
            if sel.any(): s[:] = rs[sel].mean()
        else:
            for h in range(HMIN, 24):
                dist = np.sqrt(np.nanmean((R[:, :, :h + 1] - qz[None, :, :h + 1]) ** 2, axis=(1, 2))); sel = dist <= THR
                if sel.any(): s[h] = rs[sel].mean()
        out[k] = s
    return out


def fit(trn, seed):
    e = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=250, max_features=1., n_jobs=8, random_state=seed))
    e.fit(trn[M.FULL_R3], trn.sub_ec); l = M.lg(seed); l.set_params(n_jobs=8); l.fit(trn[M.BASE_R3], trn.sub_ec)
    return e, l


def pred(m, frame, lo, hi):
    e, l = m
    p = (.6 * e.predict(frame[M.FULL_R3]) + .3 * l.predict(frame[M.BASE_R3])) / .9
    return np.clip(M.shrink(p, frame), lo, hi)


def main():
    t0 = time.time(); rows = []
    for name, i, vd, lock in folds():
        vrec = sorted(k for k in vd if k in set(ALL))
        if not any(k[1] >= 179 for k in vrec): continue
        forb = {(f, d + j) for f, d in vd | lock for j in (-1, 0, 1)}
        key = list(zip(tr.farm, tr.day.astype(int)))
        trm = np.array([k not in forb for k in key]); vam = np.array([k in set(vrec) for k in key])
        tdays = tr[trm][["farm", "day"]].drop_duplicates(); qd = pd.DataFrame(vrec, columns=["farm", "day"])
        for arm, mp in (("BASE", mapping), ("SF1", common_mapping)):
            ts, qs, notes = mp(tdays, qd, VEC)
            trn = tr[trm].copy(); va = tr[vam].copy()
            trn["season"] = [ts[(f, int(d))] for f, d in zip(trn.farm, trn.day)]
            qmap = dict(zip(vrec, qs)); va["season"] = [qmap[(f, int(d))] for f, d in zip(va.farm, va.day)]
            lo, hi = trn.sub_ec.min(), trn.sub_ec.max()
            for seed in SEEDS:
                m = fit(trn, seed)
                rows.append(pd.DataFrame(dict(validator=name, fold=i, seed=seed, arm=arm, row_id=va.row_id.values, farm=va.farm.values,
                                              day=va.day.values, hour=va.hour.values, season=va.season.values, y=va.sub_ec.values, p=pred(m, va, lo, hi))))
        print("%s fold %s: %d val records | %.0fs" % (name, i, len(vrec), time.time() - t0), flush=True)
    O = pd.concat(rows, ignore_index=True); O.to_csv(OUT / "sf1_cv_oof.csv", index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261221); res = []
    for (v, sd_), G in O.groupby(["validator", "seed"]):
        B = G[G.arm == "BASE"].set_index("row_id"); S_ = G[G.arm == "SF1"].set_index("row_id").reindex(B.index)
        p2 = B.day >= 179
        d2 = r((S_.p - S_.y)[p2]) / r((B.p - B.y)[p2]) - 1; da = r(S_.p - S_.y) / r(B.p - B.y) - 1
        per = ((S_.p - S_.y) ** 2 - (B.p - B.y) ** 2)[p2].groupby(B.farm[p2] + B.day[p2].astype(str)).sum()
        idx = rng.integers(0, len(per), (4000, len(per))); pw = float((per.values[idx].sum(1) >= 0).mean())
        res.append(dict(validator=v, seed=sd_, pass2=100 * d2, all=100 * da, P_worse=pw, n_rec2=len(per)))
    R = pd.DataFrame(res); print(R.round(3).to_string(index=False)); R.to_csv(OUT / "sf1_cv_summary.csv", index=False)
    ok = (R.pass2 < 0).all() and (R["all"] < 0).all() and (R[R.validator == "DIAG10"].P_worse < .025).all()
    print("VERDICT SF1:", "PASS" if ok else "FAIL")


def summarize(O, S):
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    print("\nseason coverage (pass-2 val records): SE1 matched rows %.1f%%, SE1full matched records %d / %d" % (
        100 * S.SE1_nh.sum() / (24 * len(S)), int((S.SE1full_nh > 0).sum()), len(S)))
    res = []; rng = np.random.default_rng(20261011); passes = True
    for (v, seed), g in O.groupby(["validator", "seed"]):
        P = g.pivot_table(index="row_id", columns="arm", values="p"); Y = g[g.arm == "BASE"].set_index("row_id").y.loc[P.index]
        rec = P.index.str[:7]
        line = {"validator": v, "seed": seed, "n_rec": rec.nunique(), "BASE": r(P.BASE - Y)}
        for a in ("SE1", "SE1p1", "SE1full"):
            line[a] = r(P[a] - Y); line[a + "_%"] = 100 * (line[a] / line["BASE"] - 1)
        # paired record bootstrap for SE1
        E = pd.DataFrame({"rec": rec, "b": (P.BASE - Y) ** 2, "s": (P.SE1 - Y) ** 2}).groupby("rec").agg(b=("b", "sum"), s=("s", "sum"), n=("b", "size"))
        idx = rng.integers(0, len(E), size=(4000, len(E))); bb = E.b.values[idx].sum(1); ss = E.s.values[idx].sum(1); nn = E.n.values[idx].sum(1)
        line["P_worse"] = float(np.mean(np.sqrt(ss / nn) >= np.sqrt(bb / nn)))
        line["recs_changed"] = int((E.b != E.s).sum())
        res.append(line)
    R = pd.DataFrame(res); pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False)); R.to_csv(OUT / "se1_cv_summary.csv", index=False)
    all_better = bool((R.SE1 < R.BASE).all()); diag_p = bool((R[R.validator == "DIAG10"].P_worse < .025).all())
    print("\nPASS rule: all 9 cells better = %s ; DIAG10 P_worse < .025 every seed = %s  ->  %s" % (
        all_better, diag_p, "PASS" if all_better and diag_p else "FAIL"))


if __name__ == "__main__":
    main()
