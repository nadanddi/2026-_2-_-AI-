# -*- coding: utf-8 -*-
"""HX2: ONE confirmation run of HX1-U (train without 'unexplained' high-EC days), corrected after the HX1 result
critique.  (2026-10-09 집 클로드, user: "고친 확인 실험 돌려봐").  Fixed before running; decided by this rule
regardless of outcome (EC pass-2-only protocol, 6.279).

Fixes vs HX1:
  * UNEX2 is defined from LEAK-FREE out-of-fold predictions (FX1 v2 configuration P, DIAG10, seeds 47/1414/6464,
    other farm's d-3..d+3 excluded): HIGH (day-mean EC >= 1.2) and day-mean (label - prediction) >= 0.3.
    -> 17 days (HX1's 14 + F13 153, F47 151, F47 154); the 5 pass-2 days are unchanged.
  * pass-2-only: the candidate is applied ONLY to pass-2 rows (day >= 179); pass-1 rows keep BASE, so only
    pass-2 rows are judged.
  * NEW seeds 3131 / 5252 / 7373 (never used in EC work) and a FRESH layout never used before: pass-2 labelled
    days per farm in 3-day chunks dealt round-robin into 6 folds.  BASE refitted with the same seeds/folds.
  * both NORMAL rows (day not in HIGH) and ALL pass-2 rows are pre-declared and reported.
Configurations: BASE (all training days) vs U2 (training without UNEX2 days); current R3 recipe
(.6 ET + .3 LGB-tweedie + .1 MLP, season DC4 + DP1).  Exclusions: same farm +-1, lock-40 +-1, other farm d-3..d+3.
Sets (pass-2 rows only): DIAG10 (folds holding pass-2 days), FRESH, EL1.
VERDICT (k = 1, alpha .025):
  PASS-IF-NO-SPIKES iff on NORMAL pass-2 rows
    (a) every new seed x {DIAG10, FRESH, EL1}: U2 better than BASE (9/9);
    (b) FRESH seed-mean (farm, day // 5) block bootstrap share(U2 not better) < .025 (20,000 draws);
    (c) sensitivity on FRESH: seed-mean U2 still better after removing (i) the 4 days that carried HX1
        (F13 231, F13 233, F47 216, F47 241) and, separately, (ii) the 4 days with the largest BASE-minus-U2
        squared-error gain in this run.
  Otherwise FAIL.  ALL-rows result is reported next to it: if ALL rows get worse, the gain exists only if the
  test has no such spike days (a hypothesis this run cannot test).
No submission files.  Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u hx2_unexplained_highec_confirm_v1.py {1|2|sum}
  stage 1 = DIAG10 + FRESH, stage 2 = EL1.
"""
import env  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = (3131, 5252, 7373)
CK = os.path.join(env.LOCAL, "hx2_ckpt")
W = (.6, .3, .1)
CONFIGS = ("BASE", "U2")
OTHER = {"F13": "F47", "F47": "F13"}
HX1_DAYS = {("F13", 231), ("F13", 233), ("F47", 216), ("F47", 241)}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def day_sets():
    ck = os.path.join(env.LOCAL, "fx1v2_ckpt")
    G = pd.concat([pd.read_csv(os.path.join(ck, f)) for f in sorted(os.listdir(ck)) if f.startswith("DIAG10")], ignore_index=True)
    G["p"] = np.mean([np.clip(W[0] * G["P_et_%d" % s] + W[1] * G["P_lgb_%d" % s] + W[2] * G["P_mlp_%d" % s], G.lo, G.hi).to_numpy()
                      for s in (47, 1414, 6464)], axis=0)
    D = G.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean"))
    assert len(D) == 360
    high = {(f, int(d)) for (f, d), v in D.y.items() if v >= 1.2}
    unex = {(f, int(d)) for (f, d), row in D.iterrows() if row.y >= 1.2 and row.y - row.p >= .3}
    assert len(high) == 26 and len(unex) == 17, (len(high), len(unex))
    return high, unex


def build_folds(lab, fds):
    p2 = {f: sorted(int(d) for d in lab[(lab.farm == f) & (lab.day >= 179)].day.unique()) for f in ("F13", "F47")}
    if STAGE == "1":
        out = [(n, i, vd) for n, i, vd in fds if n == "DIAG10" and any(d >= 179 for _, d in vd)]
        fresh = [set() for _ in range(6)]
        for f in ("F13", "F47"):
            for j in range(0, len(p2[f]), 3):
                for d in p2[f][j:j + 3]:
                    fresh[(j // 3) % 6].add((f, d))
        out += [("FRESH", 100 + k, vd) for k, vd in enumerate(fresh)]
        return out
    out = []
    for f in ("F13", "F47"):
        for k in range(0, len(p2[f]), 5):
            out.append(("EL1", 200 + len(out), {(f, d) for d in p2[f][k:k + 5]}))
    return out


def main():
    os.makedirs(CK, exist_ok=True)
    high, unex = day_sets()
    with open(os.path.join(env.LOCAL, "hx2_day_sets.json"), "w", encoding="utf-8") as fh:
        json.dump({"HIGH": sorted(map(list, high)), "UNEX2": sorted(map(list, unex))}, fh)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    for name, i, vd in build_folds(lab, fds):
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)}
                | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)}
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
        keep = {"BASE": np.ones(len(tr), bool), "U2": np.array([(f, int(d)) not in unex for f, d in zip(tr.farm, tr.day)])}
        for s in SEEDS:
            for c in CONFIGS:
                e, l, m = wt0.members(tr[keep[c]], va, s, FS, BS)
                for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                    frame["%s_%s_%d" % (c, nm, s)] = core.shrink(v, va)
            print("%s/%d seed %d" % (name, i, s), flush=True)
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
    high, unex = day_sets()
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    G = G[G.day >= 179].copy()
    nf = G.groupby("validator").validation_fold.nunique().to_dict(); print("folds:", nf)
    G["is_high"] = [(f, d) in high for f, d in zip(G.farm, G.day)]

    def pred(g, c, s):
        return np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                       g.lo, g.hi).to_numpy(float)
    cells, keep = [], {}
    for rows in ("NORMAL", "ALL"):
        for v in ("DIAG10", "FRESH", "EL1"):
            g = G[G.validator == v]
            if rows == "NORMAL":
                g = g[~g.is_high]
            if g.empty:
                continue
            y = g.sub_ec.to_numpy(float)
            sr = {c: [r(pred(g, c, s) - y) for s in SEEDS] for c in CONFIGS}
            mb, mu = np.mean([pred(g, "BASE", s) for s in SEEDS], 0), np.mean([pred(g, "U2", s) for s in SEEDS], 0)
            d = r(mu - y) / r(mb - y) - 1
            print("%-6s %-6s pass-2 days %2d  BASE %.4f [%s]  U2 %.4f [%s]  %+.2f%%" % (
                rows, v, g[["farm", "day"]].drop_duplicates().shape[0], r(mb - y), " ".join("%.4f" % x for x in sr["BASE"]),
                r(mu - y), " ".join("%.4f" % x for x in sr["U2"]), 100 * d))
            if rows == "NORMAL":
                cells += [a < b for a, b in zip(sr["U2"], sr["BASE"])]
            if rows == "NORMAL" and v == "FRESH":
                keep = dict(g=g, y=y, mb=mb, mu=mu)
    complete = nf.get("FRESH") == 6 and nf.get("EL1") == 10 and nf.get("DIAG10", 0) >= 1
    if not complete or not keep:
        print("INCOMPLETE - no verdict"); return
    g, y, mb, mu = keep["g"], keep["y"], keep["mb"], keep["mu"]
    share = boot_share(g, (mb - y) ** 2, (mu - y) ** 2)
    gain = pd.Series((mb - y) ** 2 - (mu - y) ** 2).groupby(list(zip(g.farm, g.day))).sum().sort_values(ascending=False)
    top4 = set(gain.index[:4])
    sens = {}
    for nm, drop in (("HX1 4 days", HX1_DAYS), ("top-4 gain days %s" % sorted(top4), top4)):
        m = np.array([(f, d) not in drop for f, d in zip(g.farm, g.day)])
        sens[nm] = r(mu[m] - y[m]) < r(mb[m] - y[m])
        print("   sensitivity without %s: BASE %.4f U2 %.4f -> %s" % (nm, r(mb[m] - y[m]), r(mu[m] - y[m]), "U2 better" if sens[nm] else "U2 NOT better"))
    ok = len(cells) == 9 and all(cells) and share < .025 and all(sens.values())
    print("\nVERDICT: seed x set NORMAL better %d/%d, FRESH share %.4f, sensitivity %s -> %s" % (
        sum(cells), len(cells), share, all(sens.values()), "PASS-IF-NO-SPIKES" if ok else "FAIL"))


if __name__ == "__main__":
    if STAGE != "sum":
        main()
    summarize()
