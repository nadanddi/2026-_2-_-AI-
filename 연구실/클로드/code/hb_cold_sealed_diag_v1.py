# -*- coding: utf-8 -*-
"""H-B diagnosis: is G_C2 too warm on cold, unventilated rows in EVERY cold validator?
(lab Claude, 2026-09-30; from gc2_error_conditions_v1 / catalog 6b.15)
Diagnosis only - no candidate is fitted or selected here.

Regime (input-only, row level, legal): current act_vent == 0 ("sealed") vs > 0,
crossed with current in_temp bands.  Also the row's hour (night 0-7 / day 8-15 /
evening 16-23) for the diurnal-amplitude part of H-B.
Validators: DIAG10, EXT8, EXT10, EXT12 (saved OOFs).  EXT8 has no saved TabPFN
member: there G_C2 is approximated with TabPFN := base (exact where in_temp <= 8,
where the TabPFN weight is 0) and marked "EXT8*".
Per cell: rows, G_C2 bias (pred - label) and RMSE, member biases (base, Codex,
TabPFN), for all rows and per farm.

Output: logs/hb_cold_sealed_diag_v1.log
Run:  PYTHONPATH="" <python> -u hb_cold_sealed_diag_v1.py   (from 연구실/클로드/code)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import load  # noqa: E402

BANDS = [-99, 8, 10, 12, 15, 99]


def members(split, z, z8):
    if split == "EXT8":
        base = np.mean([z8["EXT8__MASK__7"], z8["EXT8__MASK__101"]], axis=0)
        cx = np.mean([z8["EXT8__CODEX__726"], z8["EXT8__CODEX__727"]], axis=0)
        return base, cx, base.copy()
    base = np.mean([z["%s__MASK__7" % split], z["%s__MASK__101" % split]], axis=0)
    cx = np.mean([z["%s__CODEX__726" % split], z["%s__CODEX__727" % split]], axis=0)
    return base, cx, np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % split).mean(0)


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    z8 = np.load(env.LOCAL + "/temp_ext8_oof.npz", allow_pickle=True)
    for f in (z, z8):
        assert (f["row_id"] == lab.row_id.values).all()
    y = lab.sub_temp.values
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    sealed = np.where(lab.act_vent.fillna(0).values <= 0, "sealed", "vent")
    band = pd.cut(lab.in_temp, BANDS).astype(str).values
    part = np.select([lab.hour < 8, lab.hour < 16], ["night0-7", "day8-15"], "eve16-23")

    for split in ("DIAG10", "EXT8", "EXT10", "EXT12"):
        base, cx, pfn = members(split, z, z8)
        pred = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
        ok = ~np.isnan(pred)
        d = pd.DataFrame({"farm": lab.farm.values, "reg": sealed, "band": band, "part": part,
                          "e": pred - y, "eb": base - y, "ec": cx - y, "ep": pfn - y})[ok]
        name = split + ("*" if split == "EXT8" else "")
        print("=" * 110)
        print("%s  rows %d  G_C2 RMSE %.3f  bias %+.3f  | sealed share %.0f%%"
              % (name, len(d), np.sqrt((d.e ** 2).mean()), d.e.mean(), 100 * (d.reg == "sealed").mean()))
        print("  %-8s %-12s %5s | G_C2 bias  RMSE | base   codex  tabpfn | F13 bias  F47 bias" % ("regime", "in_temp", "n"))
        for (r, b), grp in d.groupby(["reg", "band"]):
            if len(grp) < 30:
                continue
            fb = {f: grp.e[grp.farm == f].mean() for f in ("F13", "F47")}
            print("  %-8s %-12s %5d | %+.3f  %.3f | %+.3f %+.3f %+.3f | %+.3f   %+.3f"
                  % (r, b, len(grp), grp.e.mean(), np.sqrt((grp.e ** 2).mean()), grp.eb.mean(), grp.ec.mean(),
                     grp.ep.mean(), fb["F13"], fb["F47"]))
        cold = d[lab.in_temp.values[ok] <= 12]
        print("  cold rows (in_temp <= 12) by regime x part of day:  bias (n)")
        for r in ("sealed", "vent"):
            s = cold[cold.reg == r]
            print("    %-7s " % r + "  ".join("%s %+.3f (%d)" % (p, s.e[s.part == p].mean(), (s.part == p).sum())
                                             for p in ("night0-7", "day8-15", "eve16-23")))


if __name__ == "__main__":
    main()
