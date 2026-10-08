# -*- coding: utf-8 -*-
"""NS1: remove ONLY the 'no-signal' unexplained high-EC days (generated-data candidates) from training.
(2026-10-09 집 클로드, user: "그 날들만 빼는 실험도 해봐"; hypothesis: generated high-EC days disturb even the
well-predicted ones).  Fixed before running.
NOSIG = UNEX2 days that showed NO input change vs neighbouring normal days in EH1 (no |z| >= 2.5):
        F47 132, F47 145, F47 154, F47 229  (4 days; 229 is pass 2).  Fixed from eh1 log before this run.
Configurations: BASE (all days) vs NS (training without the 4 NOSIG days); current R3 recipe (.6 ET + .3 LGB-tweedie +
.1 MLP, season DC4 + DP1, shrink + clip); NEW seeds 1717 / 3939 / 5757.  Exclusions same farm +-1, lock-40 +-1,
other farm d-3..d+3.  Validation rows are never dropped (the 4 days stay in scoring).
Folds: DIAG10 / A / B (judged, ALL rows); EL1 and P2LOO pass-2 (guard).
Rule (EC rule, k = 1): PASS iff every seed better on DIAG10, A, B (9/9) AND DIAG10 seed-mean (farm, day // 5) block
bootstrap share(NS not better) < .025.  GUARD: EL1 or P2LOO pass-2 worse by >= 2 % -> hold.
Reported: EXPL (9 well-predicted high days), other UNEX2 (13), NOSIG (4), normal days.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ns1_remove_nosignal_highec_v1.py {1|2|sum}
"""
import env  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ct1", os.path.join(HERE, "ct1_thermal_schedule_features_v1.py"))
ct1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct1)
wt0, dp1, dc4, p3, core = ct1.wt0, ct1.dp1, ct1.dc4, ct1.p3, ct1.core
SEEDS = (1717, 3939, 5757)
CK = os.path.join(env.LOCAL, "ns1_ckpt")
W = (.6, .3, .1)
OTHER = {"F13": "F47", "F47": "F13"}
NOSIG = {("F47", 132), ("F47", 145), ("F47", 154), ("F47", 229)}
CONFIGS = ("BASE", "NS")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def main():
    os.makedirs(CK, exist_ok=True)
    ct1.STAGE = STAGE
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    for name, i, vd in ct1.build_folds(lab, fds):
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
        keep = {"BASE": np.ones(len(tr), bool), "NS": np.array([(f, int(d)) not in NOSIG for f, d in zip(tr.farm, tr.day)])}
        for s in SEEDS:
            for c in CONFIGS:
                e, l, m = wt0.members(tr[keep[c]], va, s, FS, BS)
                for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                    frame["%s_%s_%d" % (c, nm, s)] = core.shrink(v, va)
        assert frame.notna().all().all()
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def summarize():
    sets = json.load(open(os.path.join(env.LOCAL, "hx2_day_sets.json"), encoding="utf-8"))
    high = {(f, int(d)) for f, d in sets["HIGH"]}; unex = {(f, int(d)) for f, d in sets["UNEX2"]}
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    nf = G.groupby("validator").validation_fold.nunique().to_dict(); print("folds:", nf)
    k = list(zip(G.farm, G.day))
    G["grp"] = ["NOSIG" if x in NOSIG else "UNEX(other)" if x in unex else "EXPL" if x in high else "normal" for x in k]

    def pred(g, c, s):
        return np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                       g.lo, g.hi).to_numpy(float)
    cells, share, guard = [], None, []
    views = [("DIAG10", "all"), ("A", "all"), ("B", "all"), ("EL1", "pass2"), ("P2LOO", "pass2"),
             ("DIAG10", "EXPL"), ("DIAG10", "UNEX(other)"), ("DIAG10", "NOSIG"), ("DIAG10", "normal")]
    for v, part in views:
        g = G[G.validator == v]
        if part == "pass2": g = g[g.day >= 179]
        if part in ("EXPL", "UNEX(other)", "NOSIG", "normal"): g = g[g.grp == part]
        if g.empty:
            continue
        y = g.sub_ec.to_numpy(float)
        sr = {c: [r(pred(g, c, s) - y) for s in SEEDS] for c in CONFIGS}
        m = {c: np.mean([pred(g, c, s) for s in SEEDS], 0) for c in CONFIGS}
        d = r(m["NS"] - y) / r(m["BASE"] - y) - 1
        print("%-6s %-12s days %3d  BASE %.4f (bias %+.3f)  NS %.4f (bias %+.3f)  %+.2f%%  seeds better %d/3" % (
            v, part, g[["farm", "day"]].drop_duplicates().shape[0], r(m["BASE"] - y), np.mean(m["BASE"] - y),
            r(m["NS"] - y), np.mean(m["NS"] - y), 100 * d, sum(a < b for a, b in zip(sr["NS"], sr["BASE"]))))
        if part == "all":
            cells += [a < b for a, b in zip(sr["NS"], sr["BASE"])]
        if (v, part) == ("DIAG10", "all"):
            share = ct1.boot_share(g, (m["BASE"] - y) ** 2, (m["NS"] - y) ** 2)
        if v in ("EL1", "P2LOO") and d >= .02:
            guard.append(v)
    complete = nf.get("DIAG10") == 10 and nf.get("A") == 5 and nf.get("B") == 5
    if not complete or share is None:
        print("INCOMPLETE - no verdict"); return
    ok = len(cells) == 9 and all(cells) and share < .025
    print("\nVERDICT: seed x {DIAG10,A,B} better %d/%d, DIAG10 share %.4f -> %s%s" % (
        sum(cells), len(cells), share, "PASS" if ok else "FAIL",
        ("  GUARD HOLD: %s" % guard) if guard else ("" if nf.get("P2LOO") == 46 else "  (guard sets not finished)")))


if __name__ == "__main__":
    if STAGE != "sum":
        main()
    summarize()
