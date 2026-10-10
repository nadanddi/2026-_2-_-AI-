# -*- coding: utf-8 -*-
"""TG1 (SCREEN, 2026-10-11 집 클로드, user: "이번엔 온도 모델 방향도 생각해보자"): SG2-style neighbour correction for
TEMPERATURE.  SG2 (EC, catalog 6.310) is the only pass-2 day-level correction that ever passed for EC: find labelled training
records of the same farm within +-3 calendar days whose hour-causal control signature (13 features, hours 0..h) is most
similar, and move the prediction toward their level.  Never tried for temperature (catalog grep 2026-10-11).  Temperature pass-2
error is 77 % day-level, like EC.  Fixed before running.
Quantity (model-free, from labels only): offset over hours 0..h, o(h) = mean_{t<=h}(sub_temp_t - in_temp_t).
  candidate c (labelled, not in the validation fold): o_c(h);  target: o_p(h) = mean_{t<=h}(pred_t - in_temp_t) of W40G-S.
  TG1a (hard, SG2 analogue): top-1 candidate; if |o_c - o_p| <= 1.0 C: pred = p + 0.5 (o_c - o_p)
  TG1b (soft): w_c = exp(-(dist_c - min)/0.25), o = sum w o_c;  pred = p + 0.5 exp(-|o - o_p|/1.0) (o - o_p)
Candidates, calendar, signature distance and hour-causality exactly as SG2's research code (hg3.correction path).
Pass-2 rows (day >= 179) only; scored on pass-2 rows, rough days excluded.
Stored members of 8 layouts (no refit): DIAG10, EL1 (tt1), F, G, H, I, J, K (tt12/13/17/23/24/26), 2 BASE seeds x 2 PFN
families each.  EXT10/EXT12 (tt1) reported as cold safety.
PROMOTE (screen; k = 2) iff a variant is better in every seed x family cell of all 8 layouts AND pooled (DIAG10 + F..K) farm x
5-day block bootstrap P(worse) < .0125.  Promotion needs a confirmation on a NEW layout with new seeds (pass-2 protocol).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tg1_signature_neighbor_offset_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("hg3", os.path.join(HERE, "ec3_HG3_strict_guard_gate_v1.py")); hg3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg3)
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
CK = os.path.join(env.LOCAL, "tg1_ckpt"); ALPHA = .0125; TAU = .25; GU = 1.0
LAY = {"DIAG10": ("tt1_ckpt", "DIAG10_", ("base_REF_7", "base_REF_101"), "codex_REF"),
       "EL1": ("tt1_ckpt", "EL1_", ("base_REF_7", "base_REF_101"), "codex_REF"),
       "EXT10": ("tt1_ckpt", "EXT10_", ("base_REF_7", "base_REF_101"), "codex_REF"),
       "EXT12": ("tt1_ckpt", "EXT12_", ("base_REF_7", "base_REF_101"), "codex_REF"),
       "F": ("tt12_ckpt", "fresh_", ("base_202", "base_303"), "codex_R"),
       "G": ("tt13_ckpt", "G_", ("base_R_404", "base_R_505"), "codex_R"),
       "H": ("tt17_ckpt", "H_", ("base_R_606", "base_R_707"), "codex_R"),
       "I": ("tt23_ckpt", "I_", ("base_R_808", "base_R_909"), "codex_R"),
       "J": ("tt24_ckpt", "J_", ("base_R_1010", "base_R_1111"), "codex_R"),
       "K": ("tt26_ckpt", "K_", ("base_R_1212", "base_R_1313"), "codex_R")}
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def labels():
    tx = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"])
    ty = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
    a = tx.merge(ty, on="row_id"); a = a[a.row_id.str[:3].isin(["F13", "F47"])].copy()
    a["farm"], a["day"], a["hour"] = a.row_id.str[:3], a.row_id.str[4:7].astype(int), a.row_id.str[8:10].astype(int)
    a = a.sort_values(["farm", "day", "hour"])
    a["off"] = a.sub_temp - a.in_temp
    a["coff"] = a.groupby(["farm", "day"]).off.transform(lambda s: s.expanding().mean())   # nanmean over 0..h
    return a.set_index(["farm", "day", "hour"]).coff, set(zip(a.farm, a.day))


def correct(frame, vd, COFF, labset, R, WV, hrs, SIG, ref, cal):
    A = frame.p.values.copy(); B = frame.p.values.copy()
    roles = R.set_index(["farm", "day"]).role; full_cache = {}

    def twin_date(f, d, h):
        E = [e for e in R[R.farm == f].day if (f, e) in ref]
        cols = np.asarray(hrs <= h)
        Am = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((Am - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None

    def full_date(f, d):
        if (f, d) in ref:
            return cal[(f, d)]
        if (f, d) in full_cache:
            return full_cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            days = sorted(R[R.farm == f].day); i = days.index(d)
            t = (full_date(f, days[i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        full_cache[(f, d)] = t
        return t

    for (f, d), idx in frame.groupby(["farm", "day"]).groups.items():
        if d < 179:
            continue
        G = R[R.farm == f]
        G = G[[(f, e) in ref and (f, e) in labset for e in G.day]]
        calc = np.array([cal[(f, e)] for e in G.day])
        days = sorted(R[R.farm == f].day); i = days.index(d)
        rows = frame.loc[idx].sort_values("hour")
        op = (rows.p - rows.in_temp).expanding().mean().values
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            if not np.isfinite(op[k]):
                continue
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None:
                cq = full_date(f, days[i - 1]) + 0.1
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any():
                continue
            Gm = G[m]; S = SIG[h]
            RS = S.loc[[(f, e) for e in R[R.farm == f].day if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            oc = np.array([COFF.get((f, e, h), np.nan) for e in Gm.day]); ok = np.isfinite(oc)
            if not ok.any():
                continue
            dist, oc = dist[ok], oc[ok]
            j = frame.index.get_loc(ii)
            o1 = oc[np.argmin(dist)]
            if abs(o1 - op[k]) <= GU:
                A[j] = rr.p + .5 * (o1 - op[k])
            w = np.exp(-(dist - dist.min()) / TAU); w /= w.sum(); o = float(np.sum(w * oc))
            B[j] = rr.p + .5 * np.exp(-abs(o - op[k]) / GU) * (o - op[k])
    return A, B


def load_layout(L):
    d, pre, bs, cx = LAY[L]; D = os.path.join(env.LOCAL, d)
    return [(f, pd.read_csv(os.path.join(D, f))) for f in sorted(os.listdir(D)) if f.startswith(pre) and f.endswith(".csv")]


def run():
    os.makedirs(CK, exist_ok=True)
    COFF, labset = labels()
    R, WV, hrs, SIG, DM = hg3.prepare_structure()
    for L, (d, pre, bs, cx) in LAY.items():
        for fn, M in load_layout(L):
            path = os.path.join(CK, "%s__%s" % (L, fn))
            if os.path.exists(path) or not (M.day >= 179).any():
                continue
            vd = set(zip(M.farm, M.day)); ref = {x for x in labset if x not in vd}
            cal = hg3.ref_calendar(R, WV, ref)
            g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); out = M[["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy()
            for b in bs:
                for fam in "AB":
                    F = M[["farm", "day", "hour", "in_temp"]].copy()
                    F["p"] = (.4 * M[b] + (.2 + .4 * (1 - g)) * M[cx] + .4 * g * M["pfn_" + fam]).values
                    a, bb = correct(F, vd, COFF, labset, R, WV, hrs, SIG, ref, cal)
                    out["ref_%s_%s" % (b, fam)] = F.p.values; out["a_%s_%s" % (b, fam)] = a; out["b_%s_%s" % (b, fam)] = bb
            out.to_csv(path, index=False); print("  %s %s done" % (L, fn), flush=True)


def summarize():
    rng = np.random.default_rng(20261112)
    for var in "ab":
        print("\n==== TG1%s" % var); ok = True; pool = []
        for L, (d, pre, bs, cx) in LAY.items():
            fs = [f for f in sorted(os.listdir(CK)) if f.startswith(L + "__")]
            if not fs:
                continue
            M = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in fs], ignore_index=True)
            M = M[(M.day >= 179).values & np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
            y = M.sub_temp.values; cells = []
            for b in bs:
                for fam in "AB":
                    cells.append(rm(M["%s_%s_%s" % (var, b, fam)] - y) / rm(M["ref_%s_%s" % (b, fam)] - y) - 1)
            c0 = "%s_%s_A" % (var, bs[0]); r0 = "ref_%s_A" % bs[0]
            fr = [100 * (rm((M[c0] - y)[M.farm == f]) / rm((M[r0] - y)[M.farm == f]) - 1) for f in ("F13", "F47")]
            cold = M.in_temp.values <= 10
            print("  %-6s n %4d  cells %s | F13 %+.2f%% F47 %+.2f%% | in_temp<=10 %+.2f%%" % (L, len(M), " ".join("%+.2f%%" % (100 * x) for x in cells), *fr,
                  100 * (rm((M[c0] - y)[cold]) / rm((M[r0] - y)[cold]) - 1) if cold.any() else np.nan))
            if L in ("EXT10", "EXT12"):
                continue
            ok &= all(x < 0 for x in cells)
            if L != "EL1":
                pool.append((M, c0, r0))
        sm, n = [], []
        for M, c0, r0 in pool:
            cl = (M.farm + "_" + (M.day // 5).astype(str)).values
            dd = pd.Series((M[c0] - M.sub_temp) ** 2 - (M[r0] - M.sub_temp) ** 2).groupby(cl).agg(["sum", "count"]); sm.append(dd["sum"].values); n.append(dd["count"].values)
        sm, n = np.concatenate(sm), np.concatenate(n); idx = rng.integers(0, len(sm), (20000, len(sm)))
        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
        print("  pooled P(worse) %.4f -> %s" % (p, "PROMOTE" if ok and p < ALPHA else "stop"))


if __name__ == "__main__":
    if os.environ.get("TG1_SUM") != "1":
        run()
    summarize()
