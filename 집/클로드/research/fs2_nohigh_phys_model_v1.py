# -*- coding: utf-8 -*-
"""FS2: build the new EC model of the NO-HIGH world with the feature family that passed the FS1 screen (PHYS) and
judge it.  (2026-10-09 집 클로드, user: "고EC날을 빼고 데이터를 새로 만들어 낸 다음, 모델을 새로 만들어봐. 특징도
새로 발견해보고").  Fixed before running.

World: the 26 high-EC days (HX1 set HIGH) removed from the data entirely (training, validation, season, exclusions).
Configurations (R3 = .6 ET + .3 LGB-tweedie + .1 MLP, members as WT0, shrink + clip):
  REF   current features: FULL without day + season (DC4) + DP1                 (ET uses FULL set, LGB/MLP BASE set)
  PHYS  REF + the 6 PHYS features of FS1 (vpd_now, vpd_tdmean, dT_now, dT_tdmean, closed_hours_td, closed_x_rad_td)
        added to all three members
NEW seeds 8181 / 9292 / 1010 (unused; the screen used 3131).
Folds: house DIAG10 / A / B minus high days (judged); EL1 and P2LOO on the remaining pass-2 days (descriptive);
exclusions same farm +-1, lock-40 +-1, other farm d-3..d+3.
Rule (EC rule 2026-10-04, k = 1): PASS iff every seed improves on DIAG10, A and B (9/9) AND DIAG10 seed-mean
(farm, day // 5) block bootstrap share(PHYS not better) < .025.  Guard: EL1 or P2LOO pass-2 worse by >= 2 % -> hold.
Note: the screen used DIAG10 (seed 3131, LGB only), so DIAG10 is not independent of the selection; A, B, EL1, P2LOO
and the new seeds are the out-of-selection evidence.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u fs2_nohigh_phys_model_v1.py {1|2|sum}
  stage 1 = DIAG10 + A + B, stage 2 = EL1 + P2LOO.
"""
import env  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
spec = importlib.util.spec_from_file_location("fs1", os.path.join(HERE, "fs1_nohigh_feature_screen_v1.py"))
fs1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fs1)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = (8181, 9292, 1010)
CK = os.path.join(env.LOCAL, "fs2_ckpt")
W = (.6, .3, .1)
OTHER = {"F13": "F47", "F47": "F13"}
CONFIGS = ("REF", "PHYS")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def build_folds(lab, fds, high):
    if STAGE == "1":
        return [(n, i, {k for k in vd if k not in high}) for n, i, vd in fds if n in ("DIAG10", "A", "B")]
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
    high = {(f, int(d)) for f, d in json.load(open(os.path.join(env.LOCAL, "hx1_day_sets.json"), encoding="utf-8"))["HIGH"]}
    NF, fam = fs1.new_features(full)
    phys = fam["PHYS"]; assert len(phys) == 6
    lab = lab[[(f, int(d)) not in high for f, d in zip(lab.farm, lab.day)]].copy()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    lab = lab.merge(NF[["row_id"] + phys], on="row_id", how="left", validate="one_to_one").set_index(lab.index)
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    cols = {"REF": (FS, BS), "PHYS": (FS + phys, BS + phys)}
    for name, i, vd in build_folds(lab, fds, high):
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
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    nf = G.groupby("validator").validation_fold.nunique().to_dict(); print("folds:", nf)

    def pred(g, c, s):
        return np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                       g.lo, g.hi).to_numpy(float)
    cells, share, guard = [], None, []
    for v, part, lo in (("DIAG10", "all", 0), ("A", "all", 0), ("B", "all", 0), ("DIAG10", "pass2", 179),
                        ("EL1", "pass2", 179), ("P2LOO", "pass2", 179)):
        g = G[(G.validator == v) & (G.day >= lo)]
        if g.empty:
            continue
        y = g.sub_ec.to_numpy(float)
        sr = {c: [r(pred(g, c, s) - y) for s in SEEDS] for c in CONFIGS}
        m = {c: np.mean([pred(g, c, s) for s in SEEDS], 0) for c in CONFIGS}
        d = r(m["PHYS"] - y) / r(m["REF"] - y) - 1
        print("%-6s %-5s days %3d  REF %.4f [%s]  PHYS %.4f [%s]  %+.2f%%" % (v, part, g[["farm", "day"]].drop_duplicates().shape[0],
              r(m["REF"] - y), " ".join("%.4f" % x for x in sr["REF"]), r(m["PHYS"] - y), " ".join("%.4f" % x for x in sr["PHYS"]), 100 * d))
        if part == "all":
            cells += [a < b for a, b in zip(sr["PHYS"], sr["REF"])]
        if (v, part) == ("DIAG10", "all"):
            share = boot_share(g, (m["REF"] - y) ** 2, (m["PHYS"] - y) ** 2)
        if v in ("EL1", "P2LOO") and d >= .02:
            guard.append(v)
    complete = nf.get("DIAG10") == 10 and nf.get("A") == 5 and nf.get("B") == 5
    if not complete or share is None:
        print("INCOMPLETE - no verdict"); return
    ok = len(cells) == 9 and all(cells) and share < .025
    print("\nVERDICT: seed x {DIAG10,A,B} better %d/%d, DIAG10 share %.4f -> %s%s" % (
        sum(cells), len(cells), share, "PASS" if ok else "FAIL",
        ("  GUARD HOLD: %s" % guard) if guard else ("" if nf.get("P2LOO") else "  (guard sets not finished)")))


if __name__ == "__main__":
    if STAGE != "sum":
        main()
    summarize()
