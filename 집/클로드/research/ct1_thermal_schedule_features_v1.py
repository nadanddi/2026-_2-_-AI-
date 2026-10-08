# -*- coding: utf-8 -*-
"""CT1: thermal-curtain SCHEDULE features for EC (all days, high days INCLUDED).  (2026-10-09 집 클로드, after EH3/EH4:
days whose curtain ran 'night 0 / 9 h partial / 10-15 h 100 / 16 h partial / 0' were high-EC far more often; user: "해봐").
Fixed before running.  The pattern was FOUND by exploration on these labels (post-hoc); new seeds, A/B folds and the
pass-2 sets are the out-of-discovery evidence.  The exact 31.25 day mean is NOT used (it needs the whole day and was
picked after looking); features are broad and causal (same farm, same day, hours 0..h only).
New features THS (6):
  th_full_hours_td   hours so far with act_thermal >= 99.9
  th_full_run        length of the current run of act_thermal >= 99.9 ending at h
  th_night_closed    1 if every observed hour 0..min(h,8) had act_thermal <= 0.1 (curtain fully withdrawn at night)
  th_h9              act_thermal at 9 h (NaN before 9 h)
  th_h9_partial      1 if 50 <= th_h9 <= 95 (NaN before 9 h)
  th_sched_score     share of observed hours 0..h matching the template T (0 h-8 h: <= .1; 9 h and 16 h: 50-95;
                     10 h-15 h: >= 99.9; 17 h-23 h: <= .1)
Configurations: REF = current EC features (FULL without day) + season (DC4) + DP1;  THS = REF + the 6 features in all
three R3 members (.6 ET + .3 LGB-tweedie + .1 MLP), shrink + clip.  NEW seeds 2121 / 4343 / 6565.
Folds: DIAG10 / A / B (judged, ALL rows incl. high days); EL1 and P2LOO pass-2 (guard).  Exclusions same farm +-1,
lock-40 +-1, other farm d-3..d+3.
Rule (EC rule 2026-10-04, k = 1): PASS iff every seed better on DIAG10, A and B (9/9) AND DIAG10 seed-mean
(farm, day // 5) block bootstrap share(THS not better) < .025.  GUARD: EL1 or P2LOO pass-2 worse by >= 2 % -> hold.
Reported: rows on curtain-pattern days vs others, high vs normal days.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ct1_thermal_schedule_features_v1.py {1|2|sum}
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = (2121, 4343, 6565)
CK = os.path.join(env.LOCAL, "ct1_ckpt")
W = (.6, .3, .1)
OTHER = {"F13": "F47", "F47": "F13"}
CONFIGS = ("REF", "THS")
THS = ["th_full_hours_td", "th_full_run", "th_night_closed", "th_h9", "th_h9_partial", "th_sched_score"]
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def template_ok(h, v):
    if np.isnan(v):
        return np.nan
    if h <= 8 or h >= 17:
        return float(v <= .1)
    if h in (9, 16):
        return float(50 <= v <= 95)
    return float(v >= 99.9)


def thermal_features(full):
    a = full.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    out = {c: np.full(len(a), np.nan) for c in THS}
    for _, g in a.groupby(["farm", "day"], sort=False):
        idx = g.index.values; th = g.act_thermal.to_numpy(float); hrs = g.hour.to_numpy(int)
        full_cnt = run = 0; night_ok = 1.0; h9 = np.nan; match = []
        for j, (h, v) in enumerate(zip(hrs, th)):
            isfull = (not np.isnan(v)) and v >= 99.9
            full_cnt += isfull; run = run + 1 if isfull else 0
            if h <= 8 and not np.isnan(v) and v > .1:
                night_ok = 0.0
            if h == 9:
                h9 = v
            m = template_ok(h, v)
            if not np.isnan(m):
                match.append(m)
            i = idx[j]
            out["th_full_hours_td"][i] = full_cnt; out["th_full_run"][i] = run; out["th_night_closed"][i] = night_ok
            out["th_h9"][i] = h9 if h >= 9 else np.nan
            out["th_h9_partial"][i] = (float(50 <= h9 <= 95) if not np.isnan(h9) else np.nan) if h >= 9 else np.nan
            out["th_sched_score"][i] = np.mean(match) if match else np.nan
    F = pd.DataFrame(out); F.insert(0, "row_id", a.row_id.values)
    return F


def pattern_days(full):
    P = full.pivot_table(index=["farm", "day"], columns="hour", values="act_thermal")
    m = (P[list(range(10, 16))] >= 99.9).all(1) & (P[list(range(0, 9))] <= .1).all(1) & (P[list(range(18, 24))] <= .1).all(1)
    return {(f, int(d)) for (f, d), v in m.items() if v}


def build_folds(lab, fds):
    if STAGE == "1":
        return [x for x in fds if x[0] in ("DIAG10", "A", "B")]
    out = []
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            out.append(("EL1", 300 + len(out), {(f, int(d)) for d in ds[k:k + 5]}))
    for f in ("F13", "F47"):
        for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()):
            out.append(("P2LOO", 300 + len(out), {(f, int(d))}))
    return out


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    lab = lab.merge(thermal_features(full), on="row_id", how="left", validate="one_to_one").set_index(lab.index)
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    cols = {"REF": (FS, BS), "THS": (FS + THS, BS + THS)}
    for name, i, vd in build_folds(lab, fds):
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        vd = {(f, d) for f, d in vd if ((lab.farm == f) & (lab.day == d)).any()}
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)}
                | {(f, d + j) for f, d in lock for j in (-1, 0, 1)})
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        assert not (tr_m & va_m).any() and va_m.any()
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["lo"], frame["hi"] = tr.sub_ec.min(), tr.sub_ec.max()
        for s in SEEDS:
            for c in CONFIGS:
                e, l, m = wt0.members(tr, va, s, *cols[c])
                for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                    frame["%s_%s_%d" % (c, nm, s)] = core.shrink(v, va)
        assert frame.notna().all().all()
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def boot_share(G, a, b, n=20000, seed=0):
    blk = (G.farm + "_" + (G.day // 5).astype(str)).values
    keys, inv = np.unique(blk, return_inverse=True)
    sa = np.bincount(inv, weights=a, minlength=len(keys)); sb = np.bincount(inv, weights=b, minlength=len(keys))
    rng = np.random.default_rng(seed); worse = 0
    for _ in range(n // 1000):
        ix = rng.integers(0, len(keys), size=(1000, len(keys)))
        worse += int((sb[ix].sum(1) >= sa[ix].sum(1)).sum())
    return worse / n


def summarize():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    pat = pattern_days(full)
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    nf = G.groupby("validator").validation_fold.nunique().to_dict(); print("folds:", nf)
    G["dm"] = G.groupby(["validator", "validation_fold", "farm", "day"]).sub_ec.transform("mean")
    G["pat"] = [(f, d) in pat for f, d in zip(G.farm, G.day)]

    def pred(g, c, s):
        return np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                       g.lo, g.hi).to_numpy(float)
    cells, share, guard = [], None, []
    views = [("DIAG10", "all"), ("A", "all"), ("B", "all"), ("DIAG10", "pass2"), ("EL1", "pass2"), ("P2LOO", "pass2"),
             ("DIAG10", "curtain-pattern days"), ("DIAG10", "other days"), ("DIAG10", "high days"), ("DIAG10", "normal days")]
    for v, part in views:
        g = G[G.validator == v]
        if part == "pass2": g = g[g.day >= 179]
        if part == "curtain-pattern days": g = g[g.pat]
        if part == "other days": g = g[~g.pat]
        if part == "high days": g = g[g.dm >= 1.2]
        if part == "normal days": g = g[g.dm < 1.2]
        if g.empty:
            continue
        y = g.sub_ec.to_numpy(float)
        sr = {c: [r(pred(g, c, s) - y) for s in SEEDS] for c in CONFIGS}
        m = {c: np.mean([pred(g, c, s) for s in SEEDS], 0) for c in CONFIGS}
        d = r(m["THS"] - y) / r(m["REF"] - y) - 1
        print("%-6s %-20s days %3d  REF %.4f [%s]  THS %.4f [%s]  %+.2f%%" % (
            v, part, g[["farm", "day"]].drop_duplicates().shape[0], r(m["REF"] - y), " ".join("%.4f" % x for x in sr["REF"]),
            r(m["THS"] - y), " ".join("%.4f" % x for x in sr["THS"]), 100 * d))
        if part == "all":
            cells += [a < b for a, b in zip(sr["THS"], sr["REF"])]
        if (v, part) == ("DIAG10", "all"):
            share = boot_share(g, (m["REF"] - y) ** 2, (m["THS"] - y) ** 2)
        if v in ("EL1", "P2LOO") and d >= .02:
            guard.append(v)
    complete = nf.get("DIAG10") == 10 and nf.get("A") == 5 and nf.get("B") == 5
    if not complete or share is None:
        print("INCOMPLETE - no verdict"); return
    ok = len(cells) == 9 and all(cells) and share < .025
    print("\nVERDICT: seed x {DIAG10,A,B} better %d/%d, DIAG10 share %.4f -> %s%s" % (
        sum(cells), len(cells), share, "PASS" if ok else "FAIL",
        ("  GUARD HOLD: %s" % guard) if guard else ("  (guard sets complete)" if nf.get("P2LOO") == 46 and nf.get("EL1") == 10 else "  (guard sets not finished)")))


if __name__ == "__main__":
    if STAGE != "sum":
        main()
    summarize()
