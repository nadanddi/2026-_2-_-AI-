# -*- coding: utf-8 -*-
"""EH3: among the suddenly-CLOSED days (EH2 window NB, closed_h z >= 2.5), what separates days whose EC spiked
(day mean >= 1.2) from days that did not (day mean < .8)?  (2026-10-09 집 클로드, user: "밀폐된 날 중 튄날과 안튄날을
확인해봐").  Exploratory; Mann-Whitney with Benjamini-Hochberg over all features.
Features (inputs only unless marked LABEL = diagnostic only, not usable in a model):
  day summaries of EH1 (24 h means etc.), hour-0 values (in_temp, in_hum, in_co2), closed-run length in days up to d
  (consecutive previous days with closed_h >= 20), closed hours of d-1, d-2, record day, farm, pass,
  partner farm's closed_h on the same record day and d+-2 (the partner's test days are offset by 2),
  LABEL: previous record day's EC day mean (same farm), partner's EC day mean on d-2..d+2 (max).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u eh3_closed_spiked_vs_not_v1.py
"""
import env  # noqa: F401
import os, importlib.util, sys
import numpy as np, pandas as pd
import common
from scipy.stats import mannwhitneyu
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("eh1", os.path.join(HERE, "eh1_highec_input_signatures_v1.py"))
eh1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(eh1)
OTHER = {"F13": "F47", "F47": "F13"}


def main():
    tX, ty, sX = common.load_raw()
    x = tX[tX.row_id.str[:3].isin(["F13", "F47"])].copy()
    x[["farm", "day", "hour"]] = x.row_id.str.split("_", expand=True); x.day = x.day.astype(int); x.hour = x.hour.astype(int)
    y = ty[ty.row_id.str[:3].isin(["F13", "F47"])].copy()
    y[["farm", "day", "hour"]] = y.row_id.str.split("_", expand=True); y.day = y.day.astype(int)
    dm = y.groupby(["farm", "day"]).sub_ec.mean()
    Sall = eh1.day_summaries(x)                      # all input days incl. test days (inputs only)
    Z = pd.read_csv(os.path.join(env.LOCAL, "eh2_day_z.csv"), header=[0, 1], index_col=[0, 1])
    zc = Z[("NB", "closed_h")]
    closed = [k for k, v in zc.items() if pd.notna(v) and v >= 2.5]
    h0 = x[x.hour == 0].set_index(["farm", "day"])[["in_temp", "in_hum", "in_co2"]].add_suffix("_h0")
    rows = []
    for (f, d) in closed:
        if (f, d) not in dm.index:
            continue
        ec = dm[(f, d)]
        if not (ec >= 1.2 or ec < .8):
            continue
        r = dict(Sall.loc[(f, d)]); r.update(h0.loc[(f, d)].to_dict() if (f, d) in h0.index else {})
        run = 0
        while (f, d - run - 1) in Sall.index and Sall.loc[(f, d - run - 1), "closed_h"] >= 20:
            run += 1
        r["closed_run_days"] = run
        for j in (1, 2):
            r["closed_h_d-%d" % j] = Sall.loc[(f, d - j), "closed_h"] if (f, d - j) in Sall.index else np.nan
        r["record_day"] = d; r["is_F47"] = float(f == "F47"); r["pass2"] = float(d >= 179)
        pc = [Sall.loc[(OTHER[f], d + j), "closed_h"] for j in range(-2, 3) if (OTHER[f], d + j) in Sall.index]
        r["partner_closed_h_max_pm2"] = max(pc) if pc else np.nan
        r["LABEL_prev_ec"] = dm.get((f, d - 1), np.nan)
        pe = [dm[(OTHER[f], d + j)] for j in range(-2, 3) if (OTHER[f], d + j) in dm.index]
        r["LABEL_partner_ec_max_pm2"] = max(pe) if pe else np.nan
        r.update(farm=f, day=d, ec=ec, spiked=ec >= 1.2)
        rows.append(r)
    D = pd.DataFrame(rows)
    sp, no = D[D.spiked], D[~D.spiked]
    print("closed days: spiked (>=1.2) %d, not spiked (<.8) %d; F13 %d/%d, F47 %d/%d; pass-2 %d/%d" % (
        len(sp), len(no), (sp.farm == "F13").sum(), (no.farm == "F13").sum(), (sp.farm == "F47").sum(), (no.farm == "F47").sum(),
        (sp.day >= 179).sum(), (no.day >= 179).sum()))
    feats = [c for c in D.columns if c not in ("farm", "day", "ec", "spiked")]
    res = []
    for c in feats:
        a, b = sp[c].dropna(), no[c].dropna()
        if len(a) < 4 or len(b) < 4 or (a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]):
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        res.append((c, a.median(), b.median(), u / (len(a) * len(b)), p))
    R = pd.DataFrame(res, columns=["feature", "median_spiked", "median_not", "AUC", "p"]).sort_values("p")
    m = len(R); R["q_BH"] = (R.p * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    pd.set_option("display.width", 200)
    print("\n== spiked vs not among closed days (AUC > .5 = higher on spiked days), sorted by p")
    print(R.round(3).to_string(index=False))
    print("\n== listing")
    print(D.sort_values(["farm", "day"])[["farm", "day", "ec", "closed_run_days", "in_temp", "act_heating", "act_thermal",
                                           "co2_absd", "partner_closed_h_max_pm2", "LABEL_prev_ec"]].round(2).to_string(index=False))
    D.to_csv(os.path.join(env.LOCAL, "eh3_closed_days.csv"), index=False)


if __name__ == "__main__":
    main()
