# -*- coding: utf-8 -*-
"""C18 (2026-10-11 집 클로드, user: "둘 다 합친 판으로 EC열 만들어봐"): candidate EC column for a leaderboard HYPOTHESIS TEST.
EC = 9th-round EC pipeline (submission_14 stored members) with three changes, applied in this order:
  1. B4  : 0.6 R3 + 0.4 TabPFN instead of 0.8 / 0.2 (6.355, 6.420, 6.534), same shrink and clip.
  2. HG3 : SG2 + strict high-EC correction (6.329, buffered re-check 6.539).  Research correction code (hg3.correction) with all
           400 labelled days as references (reproduces the submitted SG2 exactly, max diff 0).
  3. W2  : structural rule from 6.549 (pure pass-2 window, 'same ID record number - 2 = same house, previous date').
           For an evaluation row of record r at hour h, fires iff ALL hold (causal: hours 0..h of r, earlier records, train labels):
             (a) r >= 179 and r has no pass-1 outdoor twin over hours 0..h (date not in pass 1; twin = standardised outdoor
                 RMS <= .05, as SG2);
             (b) record r-2 is labelled and has no pass-1 outdoor twin over its full day;
             (c) either r-1 exists and is an outdoor twin of r-2 (same-date pair), or r-1 does not exist (deleted record) and
                 r-3 is not an outdoor twin of r-2 (so the deleted r-1 was r-2's pair partner);
           then pred = p + 1.0 * (a - pm_h), a = EC of r-2 at 23 h, pm_h = running mean of p over hours 0..h.
Validation of W2 is tiny (6 labelled cases, 6.549) -> NOT an adopted model; this column exists to test the hypotheses on the
leaderboard (user decision).  Interpretation is fixed before submission in the explanation document.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_c18_combined_column_v1.py
"""
import os, sys, importlib.util
import env, env_extra  # noqa: F401
import numpy as np, pandas as pd
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("hg3", os.path.join(HERE, "ec3_HG3_strict_guard_gate_v1.py")); hg3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg3)
PKG = Path(r"C:\work\farmai\집\클로드\local\submission14_ec_sg2_20261005")
OUT = Path(env.LOCAL) / "c18"; OUT.mkdir(parents=True, exist_ok=True)


def shrink(p, F):
    d = F[["farm", "day", "hour"]].reset_index(drop=True).copy(); d["p"] = p; d = d.sort_values(["farm", "day", "hour"])
    avg = d.groupby(["farm", "day"]).p.transform(lambda s: s.expanding().mean()); o = np.empty(len(p)); o[d.index.values] = (.5 * d.p + .5 * avg).values
    return o


def w2(F, pcol, WV, hrs, ec23, labset, exist):
    out = F[pcol].values.copy(); fired = []

    def twin(f, a, b, h):
        cols = np.asarray(hrs <= h)
        return np.sqrt(np.nanmean((WV.loc[(f, a)].values[cols] - WV.loc[(f, b)].values[cols]) ** 2)) <= .05

    def has_p1_twin(f, d, h):
        P1 = [e for e in WV.loc[f].index if e < 179]
        cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in P1]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        return bool((np.sqrt(np.nanmean((A - b) ** 2, axis=1)) <= .05).any())

    for (f, d), idx in F.groupby(["farm", "day"]).groups.items():
        if d < 179 or (f, d - 2) not in labset or has_p1_twin(f, d - 2, 23):
            continue
        if (f, d - 1) in exist:
            ok_c = twin(f, d - 2, d - 1, 23)
        else:
            ok_c = not ((f, d - 3) in exist and twin(f, d - 2, d - 3, 23))
        if not ok_c:
            continue
        rows = F.loc[idx].sort_values("hour"); cum = rows[pcol].expanding().mean().values; a = ec23[(f, d - 2)]
        for k, (ii, rr) in enumerate(rows.iterrows()):
            if has_p1_twin(f, d, int(rr.hour)):
                continue
            j = F.index.get_loc(ii); out[j] = rr[pcol] + (a - cum[k]); fired.append((f, d, int(rr.hour)))
    return out, fired


def main():
    z = np.load(PKG / "run1" / "prediction_details.npz"); rid = z["row_id"]
    s2 = np.load(PKG / "run1" / "sg2_details.npz"); assert (s2["row_id"] == rid).all()
    F = pd.DataFrame({"row_id": rid, "farm": [r[:3] for r in rid], "day": [int(r[4:7]) for r in rid], "hour": [int(r[8:10]) for r in rid]})
    ty = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])].copy()
    ty["farm"], ty["day"], ty["hour"] = ty.row_id.str[:3], ty.row_id.str[4:7].astype(int), ty.row_id.str[8:10].astype(int)
    ec = ty.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index); ec23 = ty[ty.hour == 23].set_index(["farm", "day"]).sub_ec
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    r3, pfn = z["raw_r3_by_seed"].mean(0), z["raw_pfn_by_context"].mean(0)
    F["p14pre"] = np.clip(shrink(.8 * r3 + .2 * pfn, F), lo, hi)
    assert np.abs(F.p14pre.values - s2["pred_submission13_config"]).max() < 1e-12
    F["pb4"] = np.clip(shrink(.6 * r3 + .4 * pfn, F), lo, hi)
    Rr, WV, hrs, SIG, DM = hg3.prepare_structure()
    vd = set(zip(F.farm, F.day)); cal = hg3.ref_calendar(Rr, WV, labset); ST = hg3.s_low_table(DM, labset)
    sg14, _ = hg3.correction(F, "p14pre", vd, set(), ec, Rr, WV, hrs, SIG, labset, cal, DM, ST)
    sgb, hgb = hg3.correction(F, "pb4", vd, set(), ec, Rr, WV, hrs, SIG, labset, cal, DM, ST)
    F["ec14"] = np.clip(sg14, lo, hi); F["b4hg3"] = np.clip(hgb, lo, hi)
    assert np.abs(F.ec14.values - s2["pred_final"]).max() < 1e-12, "SG2 reproduction failed"
    exist = set(map(tuple, pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"]) for f in ("train_X.csv", "test_X.csv")])
                    .row_id.pipe(lambda s: s[s.str[:3].isin(["F13", "F47"])]).pipe(lambda s: pd.DataFrame({"f": s.str[:3], "d": s.str[4:7].astype(int)})).drop_duplicates().values))
    exist = {(f, int(d)) for f, d in exist}
    w, fired = w2(F, "b4hg3", WV, hrs, ec23, labset, exist)
    F["c18"] = np.clip(w, lo, hi)
    print("re-derived EC14 = submission_14 EC (max diff < 1e-12)")
    for a, b in (("ec14", "b4hg3"), ("b4hg3", "c18"), ("ec14", "c18")):
        d = F[b] - F[a]; print("%-6s -> %-6s: RMS %.4f mean %+.4f min %+.3f max %+.3f rows changed %d" % (a, b, np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max(), int((d.abs() > 1e-9).sum())))
    print("W2 fired on records:", sorted({(f, d) for f, d, h in fired}), "rows", len(fired))
    D = F.groupby(["farm", "day"])[["ec14", "pb4", "b4hg3", "c18"]].mean(); D["chg"] = D.c18 - D.ec14
    print(D[D.chg.abs() > .02].round(3).to_string())
    F[["row_id", "ec14", "b4hg3", "c18"]].to_csv(OUT / "c18_ec_column.csv", index=False, float_format="%.6f")
    print("wrote", OUT / "c18_ec_column.csv")


if __name__ == "__main__":
    main()
