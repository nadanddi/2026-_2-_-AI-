# -*- coding: utf-8 -*-
"""EH2: how much do the EH1 input signatures (sudden closing, CO2 roughness, fog) actually relate to high EC, over ALL
labelled days?  (2026-10-09 집 클로드, user: "그런 공통적인 데이터의 변화가 실제 고EC를 맞추는데 얼마나 관계가
있는지도 확인해봐").  Exploratory.
For every labelled F13/F47 day (400), robust z of each day summary vs the same farm's normal days (day-mean EC < .8)
in two reference windows: NB = record days d-10..d+10 (as EH1; uses later inputs -> diagnostic only) and
PAST = d-10..d-1 (causal; usable in a model).  MAD scale = all normal days of the farm.  Reference days exclude d.
Signatures: CLOSE closed_h z >= 2.5;  CO2R co2_absd z >= 2.5;  FOG act_fog z >= 2.5;  ANY = any of them.
Report per signature and window: days flagged, share of flagged days that are high (>= 1.2) / >= .8, base rates,
recall of the 26 high days, AUC of the continuous z for high vs other days, Spearman(z, day-mean EC) among flagged
days, and the current model's day-mean error on flagged days (hx1 BASE DIAG10 OOF, seed mean).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u eh2_signature_vs_highec_v1.py
"""
import env  # noqa: F401
import os, importlib.util, sys
import numpy as np, pandas as pd
import common
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("eh1", os.path.join(HERE, "eh1_highec_input_signatures_v1.py"))
eh1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(eh1)
SIG = {"CLOSE": "closed_h", "CO2R": "co2_absd", "FOG": "act_fog"}


def main():
    tX, ty, sX = common.load_raw()
    x = tX[tX.row_id.str[:3].isin(["F13", "F47"])].copy()
    x[["farm", "day", "hour"]] = x.row_id.str.split("_", expand=True); x.day = x.day.astype(int); x.hour = x.hour.astype(int)
    y = ty[ty.row_id.str[:3].isin(["F13", "F47"])].copy()
    y[["farm", "day", "hour"]] = y.row_id.str.split("_", expand=True); y.day = y.day.astype(int)
    dm = y.groupby(["farm", "day"]).sub_ec.mean()
    S = eh1.day_summaries(x).reindex(dm.index)
    normal = [k for k, v in dm.items() if v < .8]
    ck = os.path.join(env.LOCAL, "hx1_ckpt")
    G = pd.concat([pd.read_csv(os.path.join(ck, f)) for f in os.listdir(ck) if f.startswith("DIAG10")], ignore_index=True)
    G["p"] = np.mean([np.clip(.6 * G["BASE_et_%d" % s] + .3 * G["BASE_lgb_%d" % s] + .1 * G["BASE_mlp_%d" % s], G.lo, G.hi)
                      for s in (47, 1414, 6464)], axis=0)
    err = G.groupby(["farm", "day"]).apply(lambda g: (g.p - g.sub_ec).mean())
    out = {}
    for win in ("NB", "PAST"):
        Z = pd.DataFrame(index=dm.index, columns=list(SIG.values()), dtype=float)
        for f in ("F13", "F47"):
            nf = [k for k in normal if k[0] == f]
            allnorm = S.loc[nf, list(SIG.values())]
            mad = (1.4826 * (allnorm - allnorm.median()).abs().median()).replace(0, np.nan)
            mad = mad.fillna(allnorm.std())
            for k in [k for k in dm.index if k[0] == f]:
                d = k[1]
                ref = [n for n in nf if n != k and ((abs(n[1] - d) <= 10) if win == "NB" else (d - 10 <= n[1] <= d - 1))]
                if len(ref) < 3:
                    continue
                Z.loc[k] = (S.loc[k, list(SIG.values())] - S.loc[ref, list(SIG.values())].median()) / mad
        out[win] = Z
        high = (dm >= 1.2).values; mid = (dm >= .8).values; ok = Z.notna().all(1).values
        print("\n==== window %s  (days with reference %d of %d; base rate high %.1f%%, >=.8 %.1f%%)" % (
            win, ok.sum(), len(dm), 100 * high[ok].mean(), 100 * mid[ok].mean()))
        flags = {name: (Z[col] >= 2.5).values & ok for name, col in SIG.items()}
        flags["ANY"] = np.any(list(flags.values()), axis=0)
        for name, fl in flags.items():
            n = fl.sum()
            if n == 0:
                print("  %-5s flagged 0" % name); continue
            col = SIG.get(name)
            auc = roc_auc_score(high[ok], Z[col].values[ok]) if col else np.nan
            sp = spearmanr(Z[col].values[fl], dm.values[fl])[0] if col and n > 4 else np.nan
            e = err.reindex(dm.index).values
            print("  %-5s flagged %3d | high %2d (%.0f%%, lift %.1fx) | >=.8 %.0f%% | recall of 26 high %2d | AUC(z) %.2f | "
                  "rho(z,EC) among flagged %+.2f | model day error on flagged: high %+.2f, non-high %+.2f (n %d)" % (
                      name, n, (fl & high).sum(), 100 * (fl & high).sum() / n, ((fl & high).sum() / n) / high[ok].mean(),
                      100 * (fl & mid).sum() / n, (fl & high).sum(), auc, sp,
                      np.nanmean(e[fl & high]) if (fl & high).any() else np.nan,
                      np.nanmean(e[fl & ~high]) if (fl & ~high).any() else np.nan, (fl & ~high).sum()))
    pd.concat(out, axis=1).assign(ec=dm.values).to_csv(os.path.join(env.LOCAL, "eh2_day_z.csv"))


if __name__ == "__main__":
    main()
