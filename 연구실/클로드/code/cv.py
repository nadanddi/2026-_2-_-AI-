# -*- coding: utf-8 -*-
"""Block cross-validation that reproduces the competition's test geometry."""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import (load_raw, build_panel, feature_columns, make_folds,
                    split_mask, rmse, TARGET_FARMS)
from aux_prior import fit_aux_prior, apply_aux_prior

N_FOLDS = 6

PARAMS = {
    "sub_temp": dict(learning_rate=0.03, num_leaves=63, min_child_samples=40,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                     reg_lambda=1.0),
    # EC has ~400 effectively independent days -> heavy regularisation
    "sub_ec": dict(learning_rate=0.02, num_leaves=15, min_child_samples=60,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.4,
                   reg_lambda=5.0),
}
MAX_ROUNDS = 4000


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_cache.pkl")


def prepare(use_aux=True, cache=True):
    if cache and os.path.exists(CACHE):
        panel, tX, ty, sX = pd.read_pickle(CACHE)
        if ("aux_temp_prior" in panel.columns) == use_aux:
            return panel, tX, ty, sX
    tX, ty, sX = load_raw()
    panel = build_panel(tX, sX)
    if use_aux:
        model, fc = fit_aux_prior(tX, ty)
        prior = apply_aux_prior(model, fc, tX, sX)
        panel = panel.merge(prior, on="row_id", how="left")
    panel = panel.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id",
                        how="left")
    test_ids = set(sX.row_id)
    panel["is_test"] = panel.row_id.isin(test_ids)
    if cache:
        pd.to_pickle((panel, tX, ty, sX), CACHE)
    return panel, tX, ty, sX


def baselines(tr, va, target):
    """Simple reference predictions, evaluated the same way as the model."""
    out = {}
    gm = tr.groupby("farm")[target].mean()
    out["farm mean"] = va.farm.map(gm).values
    if target == "sub_temp":
        out["in_temp as-is"] = va.in_temp.values
        # per-farm linear fit on in_temp
        pred = np.zeros(len(va))
        for f in TARGET_FARMS:
            m = tr[tr.farm == f].dropna(subset=["in_temp", target])
            k = np.polyfit(m.in_temp, m[target], 1)
            sel = (va.farm == f).values
            pred[sel] = np.poly1d(k)(va.in_temp.values[sel])
        out["in_temp linear"] = pred
    else:
        # nearest labelled day of the same farm (what a lag feature could see)
        pred = np.zeros(len(va))
        for f in TARGET_FARMS:
            m = tr[(tr.farm == f)].dropna(subset=[target])
            dm = m.groupby("day")[target].mean()
            sel = (va.farm == f).values
            vd = va.day.values[sel]
            idx = np.array(dm.index)
            near = idx[np.abs(idx[None, :] - vd[:, None]).argmin(axis=1)]
            pred[sel] = dm.loc[near].values
        out["nearest labelled day"] = pred
    return out


def run(target, panel, folds, fixed_rounds=None, verbose=True):
    lab = panel[(~panel.is_test) & panel[target].notna()].reset_index(drop=True)
    fcols = feature_columns(panel, extra_drop=["is_test", "day_mod7"])
    fcols = [c for c in fcols if c != "hour"] + ["hour"]

    oof = np.full(len(lab), np.nan)
    best_iters, base_acc = [], {}
    for i, fd in enumerate(folds):
        fd = {f: d for f, d in fd.items() if f in TARGET_FARMS}
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        if len(va) == 0:
            continue
        m = lgb.LGBMRegressor(n_estimators=fixed_rounds or MAX_ROUNDS,
                              random_state=7, n_jobs=4, verbose=-1,
                              **PARAMS[target])
        if fixed_rounds:
            m.fit(tr[fcols], tr[target])
        else:
            m.fit(tr[fcols], tr[target],
                  eval_set=[(va[fcols], va[target])], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            best_iters.append(m.best_iteration_ or MAX_ROUNDS)
        p = m.predict(va[fcols])
        oof[np.where(vam)[0]] = p
        for k, v in baselines(tr, va, target).items():
            base_acc.setdefault(k, []).append((v, va[target].values))
        if verbose:
            print("   fold %d  n_tr=%5d n_va=%4d  rmse=%.4f"
                  % (i, len(tr), len(va), rmse(p, va[target])))

    got = ~np.isnan(oof)
    res = {"oof_rmse": rmse(oof[got], lab[target].values[got]),
           "n": int(got.sum()), "best_iters": best_iters}
    res["per_farm"] = {f: rmse(oof[got & (lab.farm == f).values],
                               lab[target].values[got & (lab.farm == f).values])
                       for f in TARGET_FARMS}
    res["baselines"] = {k: rmse(np.concatenate([a for a, _ in v]),
                                np.concatenate([b for _, b in v]))
                        for k, v in base_acc.items()}
    res["oof"] = oof
    res["lab"] = lab
    res["fcols"] = fcols
    return res


def main():
    use_aux = "--no-aux" not in sys.argv
    print("building features (aux prior: %s) ..." % use_aux)
    panel, tX, ty, sX = prepare(use_aux=use_aux)
    print("panel: %d rows x %d cols  (train %d / test %d)"
          % (len(panel), panel.shape[1], (~panel.is_test).sum(),
             panel.is_test.sum()))

    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    folds = make_folds(days, n_folds=N_FOLDS, seed=0)
    for i, fd in enumerate(folds):
        print("  fold %d val days: %s" % (i, {f: len(v) for f, v in fd.items()}))

    summary = {}
    for target in ["sub_temp", "sub_ec"]:
        print("\n=== %s : early-stopping pass ===" % target)
        r = run(target, panel, folds)
        bi = int(np.median(r["best_iters"])) if r["best_iters"] else 500
        print("   median best_iter = %d" % bi)
        print("\n=== %s : fixed %d rounds (unbiased) ===" % (target, bi))
        r2 = run(target, panel, folds, fixed_rounds=bi)
        summary[target] = (r2, bi)
        print("   OOF RMSE = %.4f   (n=%d)" % (r2["oof_rmse"], r2["n"]))
        print("   per farm : " + "  ".join("%s %.4f" % kv
                                           for kv in r2["per_farm"].items()))
        print("   baselines: " + "  ".join("%s %.4f" % kv
                                           for kv in r2["baselines"].items()))

    print("\n" + "=" * 58)
    print("%-10s %10s %10s %10s" % ("target", "model", "best base", "gain"))
    for t, (r, bi) in summary.items():
        bb = min(r["baselines"].values())
        print("%-10s %10.4f %10.4f %9.1f%%"
              % (t, r["oof_rmse"], bb, 100 * (1 - r["oof_rmse"] / bb)))
    print("=" * 58)

    np.save("oof_cache.npy", {t: (r["oof"], bi) for t, (r, bi) in summary.items()},
            allow_pickle=True)
    return summary


if __name__ == "__main__":
    main()
