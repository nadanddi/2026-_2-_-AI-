# -*- coding: utf-8 -*-
"""CS1: damp spike predictions by blending the current model with a 'no unexplained-spike' model, judged on ALL
pass-2 rows (spike days included).  (2026-10-09 집 클로드, after the HX2 critique: "학습 행을 빼지 않고 급등 예측을
보수적으로 낮추는 방식, 판정은 ALL 행"; user: "해봐").  Fixed before running; the blends have never been scored.

Candidates (pass-2 rows only; pass-1 rows keep BASE):  p = (1 - w) * BASE + w * U2,  w in {0.25, 0.50}
  BASE = current R3 recipe trained on all days;  U2 = same recipe trained without the 17 UNEX2 days (HX2).
Data: saved HX2 out-of-fold predictions (seeds 3131 / 5252 / 7373; sets DIAG10 pass-2, FRESH, EL1).  No refit.
PRIMARY rows: ALL pass-2 rows (46 days).  NORMAL / HIGH rows reported.
Rule (k = 2, alpha = .025 / 2 = .0125), candidate vs BASE on ALL pass-2 rows:
  (a) every seed x {DIAG10, FRESH, EL1} better (9/9);
  (b) FRESH seed-mean bootstrap over DAYS (farm, day) share(candidate not better) < .0125 (20,000 draws);
  (c) FRESH sensitivity: still better (seed mean) after removing the 4 days with the largest gain.
  PASS here = candidate only; adoption would need ONE confirmation run with new seeds and a new layout.
Also reported (descriptive): the same on HX1 checkpoints (seeds 47 / 1414 / 6464, U = 14-day set; DIAG10 and EL1).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u cs1_spike_damping_blend_v1.py
"""
import env  # noqa: F401
import os, json
import numpy as np, pandas as pd

W = (.6, .3, .1)
BLENDS = (.25, .50)
ALPHA = .025 / 2
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def load(ck):
    G = pd.concat([pd.read_csv(os.path.join(env.LOCAL, ck, f)) for f in sorted(os.listdir(os.path.join(env.LOCAL, ck)))],
                  ignore_index=True)
    return G[G.day >= 179].reset_index(drop=True)


def r3(g, c, s):
    return np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                   g.lo, g.hi).to_numpy(float)


def day_boot(g, a, b, n=20000, seed=0):
    keys, inv = np.unique((g.farm + "_" + g.day.astype(str)).values, return_inverse=True)
    sa = np.bincount(inv, weights=a, minlength=len(keys)); sb = np.bincount(inv, weights=b, minlength=len(keys))
    rng = np.random.default_rng(seed); worse = 0
    for _ in range(n // 1000):
        ix = rng.integers(0, len(keys), size=(1000, len(keys)))
        worse += int((sb[ix].sum(1) >= sa[ix].sum(1)).sum())
    return worse / n


def run(ck, sets, seeds, cand, high, primary):
    G = load(ck); G["is_high"] = [(f, d) in high for f, d in zip(G.farm, G.day)]
    out = {}
    for rows in ("ALL", "NORMAL", "HIGH"):
        for v in sets:
            g = G[G.validator == v]
            if rows == "NORMAL": g = g[~g.is_high]
            if rows == "HIGH": g = g[g.is_high]
            if g.empty: continue
            y = g.sub_ec.to_numpy(float)
            base = {s: r3(g, "BASE", s) for s in seeds}; alt = {s: r3(g, cand, s) for s in seeds}
            line = "%-6s %-6s days %2d  BASE %.4f" % (rows, v, g[["farm", "day"]].drop_duplicates().shape[0],
                                                       r(np.mean(list(base.values()), 0) - y))
            for w in BLENDS:
                sr = [r((1 - w) * base[s] + w * alt[s] - y) for s in seeds]
                br = [r(base[s] - y) for s in seeds]
                pm = np.mean([(1 - w) * base[s] + w * alt[s] for s in seeds], 0); bm = np.mean(list(base.values()), 0)
                line += " | w%.2f %.4f (%+.1f%%, seeds better %d/3)" % (w, r(pm - y), 100 * (r(pm - y) / r(bm - y) - 1),
                                                                       sum(a < b for a, b in zip(sr, br)))
                out[(rows, v, w)] = dict(cells=[a < b for a, b in zip(sr, br)], g=g, y=y, bm=bm, pm=pm)
            print(line)
    if not primary:
        return
    print()
    for w in BLENDS:
        cells = sum((out[("ALL", v, w)]["cells"] for v in sets), [])
        o = out[("ALL", "FRESH", w)]; g, y, bm, pm = o["g"], o["y"], o["bm"], o["pm"]
        share = day_boot(g, (bm - y) ** 2, (pm - y) ** 2)
        gain = pd.Series((bm - y) ** 2 - (pm - y) ** 2, index=pd.MultiIndex.from_arrays([g.farm.values, g.day.values])) \
            .groupby(level=[0, 1]).sum().sort_values(ascending=False)
        top = set(gain.index[:4]); m = np.array([(f, d) not in top for f, d in zip(g.farm, g.day)])
        sens = r(pm[m] - y[m]) < r(bm[m] - y[m])
        ok = len(cells) == 9 and all(cells) and share < ALPHA and sens
        print("w%.2f: ALL seed x set better %d/9, FRESH day-bootstrap share %.4f, without top-4 gain days %s (%.4f vs %.4f) -> %s"
              % (w, sum(cells), share, sorted(top), r(pm[m] - y[m]), r(bm[m] - y[m]), "PASS (candidate; confirmation run needed)" if ok else "FAIL"))


def main():
    s2 = json.load(open(os.path.join(env.LOCAL, "hx2_day_sets.json"), encoding="utf-8"))
    print("== PRIMARY: HX2 checkpoints (seeds 3131/5252/7373), candidate U2")
    run("hx2_ckpt", ("DIAG10", "FRESH", "EL1"), (3131, 5252, 7373), "U2", {tuple(x) for x in s2["HIGH"]}, True)
    s1 = json.load(open(os.path.join(env.LOCAL, "hx1_day_sets.json"), encoding="utf-8"))
    print("\n== descriptive: HX1 checkpoints (seeds 47/1414/6464), candidate U (14 days)")
    run("hx1_ckpt", ("DIAG10", "EL1"), (47, 1414, 6464), "U", {tuple(x) for x in s1["HIGH"]}, False)


if __name__ == "__main__":
    main()
