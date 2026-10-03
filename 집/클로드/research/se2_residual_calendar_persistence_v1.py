# -*- coding: utf-8 -*-
"""SE2 (diagnostic, fixed before running; 2026-10-03 집 클로드).
Does the remaining day-level error persist along the TRUE calendar (C6.205 date
index; pass 1 only) within a 동?  A slowly varying unobserved factor (manual
operation steps, literature B) would show residual autocorrelation over several
dates and step-like runs.  DIAG10 public OOF day residuals, temperature W40G and EC
R3S; 동 from local/st_dong_assign_v1.csv.  Per farm x 동: Spearman of residual at
date t vs the same 동's residual at the nearest earlier date within lag 1..3 / 4..7
/ 8..14 dates; run-length of same-sign residuals vs a shuffle (2000).
Reading (fixed): persistence clue if rho(lag 1..3) >= .30 in both farms for a target
(n >= 30 each)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv"))
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv")); T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
T["e"] = T.sub_temp - (0.4 * (T.mask_base_7 + T.mask_base_101) / 2 + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn)
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); E = E[E.validator == "DIAG10"].copy()
E["e"] = E.sub_ec - E[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
rng = np.random.default_rng(20261003)
for name, F in (("temp W40G", T), ("EC R3S", E)):
    R = F.groupby(["farm", "day"]).e.mean()
    print("\n[%s]" % name)
    for f in ("F13", "F47"):
        G = D[(D.farm == f) & (D.day < 179)].sort_values("day").copy()
        G["date"] = np.cumsum(G.role != "second") - 1
        G["e"] = [R.get((f, d), np.nan) for d in G.day]
        for dg in ("A", "B", "all"):
            H = G if dg == "all" else G[G.dong == dg]
            H = H.dropna(subset=["e"])
            out = []
            for lo, hi in ((1, 3), (4, 7), (8, 14)):
                x, y = [], []
                for _, r in H.iterrows():
                    prev = H[(H.date <= r.date - lo) & (H.date >= r.date - hi)]
                    if dg == "all":
                        prev = prev  # any 동
                    if len(prev):
                        x.append(prev.iloc[-1].e); y.append(r.e)
                out.append("lag%d-%d rho %.2f (n %d)" % (lo, hi, spearmanr(x, y).correlation if len(x) > 5 else np.nan, len(x)))
            s = np.sign(H.e.values)
            runs = np.mean([len(list(gr)) for _, gr in __import__("itertools").groupby(s)])
            sh = np.mean([np.mean([len(list(gr)) for _, gr in __import__("itertools").groupby(rng.permutation(s))]) for _ in range(2000)])
            print("  %s %-3s n %3d | %s | mean sign-run %.2f vs shuffle %.2f" % (f, dg, len(H), " | ".join(out), runs, sh))
