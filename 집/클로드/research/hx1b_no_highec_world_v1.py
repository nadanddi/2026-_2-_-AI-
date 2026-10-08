# -*- coding: utf-8 -*-
"""HX1b: the 26 high-EC days (HX1 set HIGH, day-mean EC >= 1.2) removed from the data ENTIRELY - from
training, from validation folds, from the season index and from the +-1 exclusion rules - as if they did not
exist.  (2026-10-09 집 클로드, user: "고EC 26일을 학습, 검증에서 전부 제외하고 다시 학습 돌려봐").
Fixed before running.  Descriptive follow-up of HX1 (no new verdict; HX1's pre-set verdict stands).

Model: current R3 (.6 ET + .3 LGB-tweedie + .1 MLP, season DC4 + DP1), seeds 47 / 1414 / 6464, one
configuration (NOHIGH).  Folds: DIAG10 (10), A (5), B (5), EL1 pass-2 (5-day groups of the remaining pass-2
days), P2LOO (each remaining pass-2 day).  Fold memberships = the house folds minus HIGH days (no re-dealing).
Exclusions: same farm +-1, lock-40 +-1, other farm d-3..d+3 (as HX1/KF1).
Reported: RMSE per validator (all rows, pass-1, pass-2) for NOHIGH, next to HX1 BASE scored on the same
rows (HX1 BASE was trained WITH high days; its fold exclusions differ slightly, stated as a limitation).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u hx1b_no_highec_world_v1.py {1|2|sum}
  stage 1 = DIAG10 + A + B, stage 2 = EL1 + P2LOO.
"""
import env  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = wt0.SEEDS
CK = os.path.join(env.LOCAL, "hx1b_ckpt")
W = (.6, .3, .1)
OTHER = {"F13": "F47", "F47": "F13"}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def high_set():
    s = json.load(open(os.path.join(env.LOCAL, "hx1_day_sets.json"), encoding="utf-8"))
    h = {(f, int(d)) for f, d in s["HIGH"]}
    assert len(h) == 26
    return h


def build_folds(lab, fds, high):
    if STAGE == "1":
        return [(n, i, {k for k in vd if k not in high}) for n, i, vd in fds if n in ("DIAG10", "A", "B")]
    folds = []
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for f in ("F13", "F47"):
        for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()):
            folds.append(("P2LOO", len(folds), {(f, int(d))}))
    return folds


def main():
    os.makedirs(CK, exist_ok=True)
    high = high_set()
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab[[(f, int(d)) not in high for f, d in zip(lab.farm, lab.day)]].copy()
    assert lab[["farm", "day"]].drop_duplicates().shape[0] == 334
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    for name, i, vd in build_folds(lab, fds, high):
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        vd = {(f, d) for f, d in vd if ((lab.farm == f) & (lab.day == d)).any()}
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
        for s in SEEDS:
            e, l, m = wt0.members(tr, va, s, FS, BS)
            for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                frame["NOHIGH_%s_%d" % (nm, s)] = core.shrink(v, va)
        assert frame.notna().all().all()
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    print("folds:", G.groupby("validator").validation_fold.nunique().to_dict())
    hx = os.path.join(env.LOCAL, "hx1_ckpt")
    B = pd.concat([pd.read_csv(os.path.join(hx, f)) for f in sorted(os.listdir(hx))], ignore_index=True)

    def pred(g, c):
        return np.mean([np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                                g.lo, g.hi).to_numpy(float) for s in SEEDS], axis=0)

    def seeds(g, c):
        return [r(np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                          g.lo, g.hi).to_numpy(float) - g.sub_ec.to_numpy(float)) for s in SEEDS]
    for v in ("DIAG10", "A", "B", "EL1", "P2LOO"):
        for part, lo, hi in (("all", 0, 999), ("pass1", 0, 179), ("pass2", 179, 999)):
            g = G[(G.validator == v) & (G.day >= lo) & (G.day < hi)]
            if g.empty:
                continue
            y = g.sub_ec.to_numpy(float)
            line = "%-6s %-5s days %3d  NOHIGH %.4f [%s]" % (v, part, g[["farm", "day"]].drop_duplicates().shape[0],
                                                          r(pred(g, "NOHIGH") - y), " ".join("%.4f" % x for x in seeds(g, "NOHIGH")))
            if v in ("DIAG10", "A", "B", "EL1"):
                b = B[(B.validator == v)][["row_id", "lo", "hi"] + [c for c in B.columns if c.startswith("BASE_")]]
                gb = g[["row_id", "sub_ec"]].merge(b, on="row_id", how="inner")
                if len(gb) == len(g):
                    yb = gb.sub_ec.to_numpy(float); rb = r(pred(gb, "BASE") - yb)
                    line += "  | HX1 BASE (trained with high days) %.4f -> %+.1f%%" % (rb, 100 * (r(pred(g, "NOHIGH") - y) / rb - 1))
            print(line)


if __name__ == "__main__":
    if STAGE != "sum":
        main()
    summarize()
