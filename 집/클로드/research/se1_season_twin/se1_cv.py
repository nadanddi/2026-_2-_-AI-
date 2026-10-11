# -*- coding: utf-8 -*-
"""SE1 (2026-10-11 집 클로드, EC season).  HYPOTHESIS AND RULE FIXED BEFORE RUNNING.

Problem (audit_ec a1/a6/a7): pass-2 (record >= 179) training records get season from a PAV curve fitted on
exact-weather-twin anchors, but validation / evaluation records get np.interp along record number.  For evaluation
records with a twin, interp differs from the twin date by 4.5 days on average (max 12.7); the model is sensitive.

SE1 (the ONE tested arm): for a pass-2 validation/evaluation row at hour h,
  if h >= 5: compare the record's OWN outdoor inputs (out_temp, out_hum, out_rad, out_wspd) at hours 0..h with every
  TRAINING record of the fold (both farms): pass-1 records (season = record day) AND pass-2 training records
  (season = the fold mapping's training value, i.e. what the model was trained with).  z-standardisation = season.py
  (mu/sd from the fold's training pass-1 weather).  distance = sqrt(nanmean over 4 vars x hours 0..h of squared z
  difference).  If any reference has distance <= .05: season = mean of their season values.
  else (h < 5 or no match): season = current interp value (BASE).
  Uses only the row's own current/earlier inputs + training data -> rule-safe.
Diagnostics only (not tested, not deployable): SE1p1 = SE1 with pass-1 references only; SE1full = full-day (hours
0..23, NON-causal) twin with SE1 references.

Model: proxy of the submitted R3 (as audit a2): ExtraTrees(250, max_features 1, FULL_R3) .6 + LightGBM (model.lg,
BASE_R3) .3, normalised /.9, then model.shrink (within-record running mean, causal) and clip to training EC range.
Season of every training fold is rebuilt with season.mapping on the TRAINING records of that fold only; validation
rows' BASE season = mapping's interp output for the validation days (no validation weather in the PAV fit).

Validators (pass-2 rows scored; training drops validation records and their +-1 same-farm neighbours):
  DIAG10 : Codex DIAG10 folds (p3.prepare, lock days and their +-1 also dropped from training, never scored).
  BLK10  : audit 10-record blocks, fold = ((day//10)*7 + 3*(farm==F47)) % 5, all 400 records.
  HOLE5  : eval geometry - every pass-2 training run cut into chunks of <= 5 consecutive records; fold j holds out
           chunk j of F13 and chunk j of F47 together (7 folds), all 400 records otherwise.
Seeds: 7, 101, 2024 (ET and LGB random_state).
PRE-FIXED PASS: SE1 pass-2 row RMSE < BASE in all 9 cells (3 validators x 3 seeds) AND DIAG10 record-level paired
bootstrap P(SE1 RMSE >= BASE RMSE) < .025 for every seed (B = 4000, resampling pass-2 records).  Size irrelevant.
Run:  cd 집/클로드/research && PYTHONPATH="" PYTHONIOENCODING=utf-8 py -3.12 -u se1_season_twin/se1_cv.py
"""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
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

OUT = Path(r"C:\work\farmai\집\클로드\research\se1_season_twin")
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
    t0 = time.time(); rows = []; seasons_log = []
    for name, i, vd, lock in folds():
        vp2 = sorted(k for k in vd if k[1] >= 179 and k in set(ALL))
        if not vp2: continue
        forb = {(f, d + j) for f, d in vd | lock for j in (-1, 0, 1)}
        key = list(zip(tr.farm, tr.day.astype(int)))
        trm = np.array([k not in forb for k in key]); vam = np.array([k in set(vp2) for k in key])
        trn = tr[trm].copy(); va = tr[vam].copy()
        tdays = trn[["farm", "day"]].drop_duplicates(); qd = pd.DataFrame(vp2, columns=["farm", "day"])
        ts, qs, notes = mapping(tdays, qd, VEC)
        mu, sd = np.array(notes["standardization"]["mu"]), np.array(notes["standardization"]["sd"])
        trn["season"] = [ts[(f, int(d))] for f, d in zip(trn.farm, trn.day)]
        base = dict(zip(vp2, qs))
        refs = sorted(ts); rseason = [ts[k] for k in refs]
        p1 = [k for k in refs if k[1] < 179]
        T = {"SE1": twin_seasons(vp2, refs, rseason, mu, sd),
             "SE1p1": twin_seasons(vp2, p1, [ts[k] for k in p1], mu, sd),
             "SE1full": twin_seasons(vp2, refs, rseason, mu, sd, full=True)}
        frames = {}
        b = np.array([base[(f, int(d))] for f, d in zip(va.farm, va.day)]); va = va.assign(season=b); frames["BASE"] = va
        for arm, tw in T.items():
            s = np.array([tw[(f, int(d))][h] for f, d, h in zip(va.farm, va.day, va.hour)])
            frames[arm] = va.assign(season=np.where(np.isfinite(s), s, b))
        for k in vp2:
            seasons_log.append(dict(validator=name, fold=i, farm=k[0], day=k[1], base=base[k],
                                    **{a + "_h23": T[a][k][23] for a in T}, **{a + "_nh": int(np.isfinite(T[a][k]).sum()) for a in T}))
        lo, hi = trn.sub_ec.min(), trn.sub_ec.max()
        for seed in SEEDS:
            m = fit(trn, seed)
            for arm, fr in frames.items():
                rows.append(pd.DataFrame(dict(validator=name, fold=i, seed=seed, arm=arm, row_id=fr.row_id.values,
                                              farm=fr.farm.values, day=fr.day.values, hour=fr.hour.values,
                                              season=fr.season.values, y=fr.sub_ec.values, p=pred(m, fr, lo, hi))))
        print("%s fold %s: %d pass-2 val records, train %d rows | %.0fs" % (name, i, len(vp2), len(trn), time.time() - t0), flush=True)
    O = pd.concat(rows, ignore_index=True); O.to_csv(OUT / "se1_cv_oof.csv", index=False)
    S = pd.DataFrame(seasons_log); S.to_csv(OUT / "se1_cv_seasons.csv", index=False)
    summarize(O, S)


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
