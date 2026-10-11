# -*- coding: utf-8 -*-
"""TH1 (2026-10-12 집 클로드): label-free same-house THERMAL HISTORY from the linked previous record's INPUTS.
Finding (truth chains, this session): x = mean in_temp of the TRUE same-house previous date over 18-23 h minus today's 0 h in_temp
correlates with the 0 h temperature residual .48 (pass 1) / .81 (pass 2) and the day residual .39 / .50, while the same quantity
from the previous record NUMBER (what the model sees; 86% a different house) has ~0.  Needs no label, so it can serve records
whose linked previous record is UNLABELLED (inside evaluation chunks: same-ID earlier evaluation record -> its inputs are allowed
'same greenhouse previous inputs').  Linker v7 never proposes other-farm evaluation records (other-farm candidates are training).
Fixed before running:
  base   = round-11 recipe q + submitted anchor rule (p>=.8, const .36, lin8, clip 1, gate 1.5) [labelled anchors only]
  S      = records whose linker v7 mirror-CV previous record is UNLABELLED, p >= p_min, not cross-ID (xf False)
  x      = mean in_temp(prev, hours W) - in_temp(rec, 0 h), clipped +-c
  beta_h = LS slope per hour of (y - base) on x, measured on PASS-1 records of S in DIAG10 (all 4 cells pooled)
  pred_h = base_h + beta_h * x  on S
  grid (reported): W {23, 18-23, 12-23, 0-23} x p_min {.5, .8, .95} x c {2, inf} = 24;  PRIMARY = (18-23, .8, inf)
Judgement of PRIMARY on pass-2: DIAG10 all 4 cells better AND record bootstrap P(worse) < .025 AND jackknife-1 same direction;
G, EL1, F, J reported (all cells same direction required for 'clean' pass).  EXT10/12 reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u th1_temp_linked_prev_thermal_history_v1.py
"""
import os, sys, itertools, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s); s.loader.exec_module(A7)
A2 = A7.A2; rm = A2.rm
WIN = {"23": [23], "18-23": list(range(18, 24)), "12-23": list(range(12, 24)), "0-23": list(range(24))}
GRID = list(itertools.product(WIN, (.5, .8, .95), (2.0, np.inf))); PRIMARY = ("18-23", .8, np.inf)


def main():
    Lall = pd.read_csv(A7.LK)
    La = Lall[Lall.prev_is_labelled.astype(bool) & Lall.anchor_st23.notna() & (Lall.steps_to_anchor == 1)].copy()
    Lu = Lall[~Lall.prev_is_labelled.astype(bool) & ~Lall.xf.astype(bool)].set_index("rec")
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])]
    X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int); TI = X.pivot_table(index="rec", columns="h", values="in_temp")
    xv = {w: pd.Series({r: TI.loc[p, hs].mean() - TI.at[r, 0] if (p in TI.index and r in TI.index) else np.nan for r, p in Lu.prev.items()}) for w, hs in WIN.items()}
    R = A2.A1.frames(); rng = np.random.default_rng(20261213)
    bases = {}
    for Ln in ("DIAG10", "G", "EL1", "F", "J", "EXT10", "EXT12"):
        M, cells = R[Ln]; key, A = A7.build(M, La)
        bases[Ln] = (M, key, cells, [A7.correct(M, q, key, A, .8, "const", "lin8", 1, 1.5) for q in cells])
    M, key, cells, base = bases["DIAG10"]; h = M.hour.values; y = M.sub_temp.values
    betas = {}
    for g in GRID:
        w, pmin, c = g; S = Lu[Lu.p >= pmin].index; x = xv[w].reindex(S).dropna().clip(-c, c)
        num = np.zeros(24); den = np.zeros(24)
        for b in base:
            for rec in x.index:
                if int(rec[4:7]) >= 179:
                    continue
                m = key == rec
                if m.any():
                    num[h[m]] += x[rec] * (y[m] - b[m]); den[h[m]] += x[rec] ** 2
        betas[g] = (num / np.maximum(den, 1e-9), x)
    print("PRIMARY beta_h: " + " ".join("%d:%.2f" % (i, v) for i, v in enumerate(betas[PRIMARY][0])), "| S records:", len(betas[PRIMARY][1]),
          "pass-2:", int((betas[PRIMARY][1].index.str[4:7].astype(int) >= 179).sum()), flush=True)

    def apply(Ln, g):
        M, key, cells, base = bases[Ln]; beta, x = betas[g]; hh = M.hour.values; outs = []
        xr = pd.Series(key).map(x).values
        for b in base:
            o = b.copy(); m = np.isfinite(xr); o[m] = b[m] + beta[hh[m]] * xr[m]; outs.append(o)
        return outs
    rows = []
    for g in GRID:
        o = apply("DIAG10", g); m2 = M.day.values >= 179
        rows.append(dict(W=g[0], p=g[1], c=g[2], pass1=100 * np.mean([rm((a - y)[~m2]) / rm((b - y)[~m2]) - 1 for a, b in zip(o, base)]),
                         F13=100 * np.mean([rm((a - y)[m2 & (M.farm.values == "F13")]) / rm((b - y)[m2 & (M.farm.values == "F13")]) - 1 for a, b in zip(o, base)]),
                         F47=100 * np.mean([rm((a - y)[m2 & (M.farm.values == "F47")]) / rm((b - y)[m2 & (M.farm.values == "F47")]) - 1 for a, b in zip(o, base)]),
                         pass2=100 * np.mean([rm((a - y)[m2]) / rm((b - y)[m2]) - 1 for a, b in zip(o, base)])))
    print(pd.DataFrame(rows).round(2).to_string(), flush=True)
    ok = True
    for Ln in ("DIAG10", "G", "EL1", "F", "J", "EXT10", "EXT12"):
        MM, kk, cc, bb = bases[Ln]; yy = MM.sub_temp.values; m2 = MM.day.values >= 179; o = apply(Ln, PRIMARY)
        rel = [rm((a - yy)[m2]) / rm((b - yy)[m2]) - 1 for a, b in zip(o, bb)]
        P = pd.concat([pd.Series(((a - yy) ** 2 - (b - yy) ** 2)[m2]).groupby(kk[m2]).sum() for a, b in zip(o, bb)], axis=1).mean(1); P = P[P.abs() > 1e-12]
        line = "  %-6s PRIMARY pass-2 cells %s | records %d improved %d" % (Ln, " ".join("%+.2f%%" % (100 * v) for v in rel), len(P), int((P < 0).sum()))
        if Ln in ("DIAG10", "G", "EL1", "F", "J"):
            ok &= all(v < 0 for v in rel)
        if Ln == "DIAG10" and len(P):
            idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean()); top = P.idxmin(); jk = P.drop(top).sum() < 0
            ok &= (pw < .025) and jk
            line += " | P(worse) %.4f | top %s share %.0f%% jackknife-1 %s" % (pw, top, 100 * P.min() / P.sum(), "same" if jk else "FLIPS")
        print(line, flush=True)
    print("VERDICT TH1 PRIMARY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
