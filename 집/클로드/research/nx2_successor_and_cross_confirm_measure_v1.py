# -*- coding: utf-8 -*-
"""NX2 (2026-10-12 집 클로드; user "1~3 전부 진행"): measure two new information channels before building rules.
(1) SUCCESSOR anchor (same house, NEXT date, training label): oracle value with label-certain truth links (TL7, a -> b means a
    is b's previous date; so b is a's successor).  For target a and each hour h: slope of the residual (y_h - q_h) on
    (S0 - q_h), S0 = b's 0 h label; reported for temperature (q = round-11 recipe DIAG10 OOF) and EC (q = RV1 CUR sg OOF).
    Large late-hour slopes would make a successor anchor worth identifying (identification itself is the next step).
(2) CROSS-TARGET CONFIRMATION of a link: for linker v7 mirror-CV choices (labelled previous record, any p), is the link more
    often correct (vs truth TL7) when the EC anchor agrees with the EC model at 0 h (|A_ec - sg_0| small) or when the temperature
    anchor agrees (|A_st - .36 - q_0| small)?  Reported as hit rate by agreement quantile, and AUC of each agreement for 'correct'.
Descriptive only (training data).  Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u nx2_successor_and_cross_confirm_measure_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)


def auc(score, lab):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def main():
    import pickle
    S3 = os.path.join(env.LOCAL, "linker_v3")
    R = pickle.load(open(os.path.join(S3, "base.pkl"), "rb")); r = R["recs"].reset_index(drop=True)
    rid = [f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)]
    TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TL = TL[TL.cert]
    succ = {rid[a]: rid[b] for a, b in zip(TL.a1, TL.b)}; pred = {rid[b]: rid[a] for a, b in zip(TL.a1, TL.b)}
    ty = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); ty["rec"] = ty.row_id.str[:7]; ty["h"] = ty.row_id.str[8:10].astype(int)
    ST = ty.pivot_table(index="rec", columns="h", values="sub_temp"); EC = ty.pivot_table(index="rec", columns="h", values="sub_ec")
    # temperature q (round-11 recipe, DIAG10 OOF, cell mean)
    FR = A2.A1.frames(); M, cells = FR["DIAG10"]; M = M.assign(q=np.mean(cells, 0), rec=M.farm + "_" + M.day.map("{:03d}".format))
    QT = M.pivot_table(index="rec", columns="hour", values="q")
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
    O["rec"] = O.row_id.str[:7]; QE = O.groupby(["rec", "hour"]).sg.mean().unstack()
    print("(1) successor oracle slope by hour (target a, successor b = same house next date, label-certain):")
    for nm, Y, Q in (("temp", ST, QT), ("EC", EC, QE)):
        recs = [a for a in succ if a in Q.index and a in Y.index and succ[a] in Y.index]
        sl = []
        for h in range(24):
            x = np.array([Y.at[succ[a], 0] - Q.at[a, h] for a in recs]); yv = np.array([Y.at[a, h] - Q.at[a, h] for a in recs])
            m = np.isfinite(x) & np.isfinite(yv); sl.append(float((x[m] * yv[m]).sum() / (x[m] ** 2).sum()))
        # compare with predecessor anchor slope at hour h (A = pred 23h)
        recp = [b for b in pred if b in Q.index and b in Y.index and pred[b] in Y.index]
        pl = []
        for h in range(24):
            x = np.array([Y.at[pred[b], 23] - Q.at[b, h] for b in recp]); yv = np.array([Y.at[b, h] - Q.at[b, h] for b in recp])
            m = np.isfinite(x) & np.isfinite(yv); pl.append(float((x[m] * yv[m]).sum() / (x[m] ** 2).sum()))
        print("  %-4s successor (n %d): %s" % (nm, len(recs), " ".join("%d:%.2f" % (h, v) for h, v in enumerate(sl) if h % 3 == 0 or h >= 20)))
        print("  %-4s predecessor(n %d): %s" % (nm, len(recp), " ".join("%d:%.2f" % (h, v) for h, v in enumerate(pl) if h % 3 == 0 or h >= 20)))
    print("(2) cross-target confirmation of linker v7 mirror-CV links:")
    L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")); L = L[L.prev_is_labelled.astype(bool) & L.rec.isin(list(pred))].copy()
    L["correct"] = [pred[a] == p for a, p in zip(L.rec, L.prev)]
    L["ag_ec"] = [abs(EC.at[p, 23] - QE.at[a, 0]) if (p in EC.index and a in QE.index) else np.nan for a, p in zip(L.rec, L.prev)]
    L["ag_t"] = [abs(ST.at[p, 23] - .36 - QT.at[a, 0]) if (p in ST.index and a in QT.index) else np.nan for a, p in zip(L.rec, L.prev)]
    L = L.dropna(subset=["ag_ec", "ag_t"]); lab = L.correct.values
    print("  links %d, correct %.2f | AUC for 'correct': linker p %.3f, EC agreement %.3f, temp agreement %.3f, p + both (rank mean) %.3f" % (
        len(L), lab.mean(), auc(L.p.values, lab), auc(-L.ag_ec.values, lab), auc(-L.ag_t.values, lab),
        auc((pd.Series(L.p).rank() + pd.Series(-L.ag_ec).rank() + pd.Series(-L.ag_t).rank()).values, lab)))
    for nm, c in (("EC agreement", "ag_ec"), ("temp agreement", "ag_t")):
        q = pd.qcut(L[c], 4, labels=["Q1(best)", "Q2", "Q3", "Q4(worst)"])
        print("  %-15s hit by quartile: %s" % (nm, " ".join("%s %.2f" % (k, v) for k, v in L.groupby(q, observed=True).correct.mean().items())))
    for pm in (.8,):
        H = L[L.p >= pm]; q = H.ag_ec <= H.ag_ec.median()
        print("  p>=%.1f (n %d, hit %.2f): EC-agree half hit %.2f vs other half %.2f; temp-agree half hit %.2f vs %.2f" % (
            pm, len(H), H.correct.mean(), H[q].correct.mean(), H[~q].correct.mean(), H[H.ag_t <= H.ag_t.median()].correct.mean(), H[H.ag_t > H.ag_t.median()].correct.mean()))


if __name__ == "__main__":
    main()
