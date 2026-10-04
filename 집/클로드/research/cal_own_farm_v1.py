# -*- coding: utf-8 -*-
"""Own-greenhouse calendar (2026-10-04 집 클로드), inputs of ONE farm only.
For a pass-2 query record, every pass-1 record and every earlier pass-2 record is an
earlier input of the same greenhouse (rule 5), so the calendar below is legal for
pass-2 rows as long as pass-2 records are dated only from records before them.
  Pass 1: records in record order; a record whose 24 h outdoor weather equals the
  previous record's (pair 'second', exact twin z-RMSE <= .05) shares its date.
  Date order = record order refined by or-opt moves (segments of 1-3 dates moved up to
  8 positions) that lower the total midnight outdoor continuity cost (deep_cal_8 cost:
  trend-corrected 23 h -> 0 h jump of out_temp / out_hum / out_wspd(.3) / out_rad,
  scaled by the farm's hourly-change SD).  A break whose cost is above the 97.5 % quantile
  of the final link costs opens a gap of GAP dates (dates missing from pass 1).
  Pass 2 (record order): exact outdoor twin among EARLIER records -> twin's date;
  otherwise previous record's date + 1 (+0 for a pair 'second').
Returns dict (farm, day) -> calendar (float)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd

W = ["out_temp", "out_hum", "out_rad", "out_wspd"]
WT = {"out_temp": 1, "out_hum": 1, "out_wspd": .3, "out_rad": 1}
GAP = 6


def load():
    X = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"] + W) for f in ("train_X.csv", "test_X.csv")])
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    roles = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
    return X, roles


def build(X, roles, farm):
    XF = X[X.farm == farm]
    piv = XF.pivot_table(index="day", columns="hour", values=W)
    days = sorted(piv.index)
    p1 = [d for d in days if d < 179]
    Z = piv.copy()
    for v in W:
        mu, sd = np.nanmean(piv.loc[p1, v].values), np.nanstd(piv.loc[p1, v].values)
        Z[v] = (piv[v] - mu) / sd
    sc = {v: np.nanstd(np.diff(piv[v].values, axis=1)) for v in W}

    def cost(a, b):
        c = 0.0
        for v in W:
            A, B = piv.loc[a, v], piv.loc[b, v]
            pg = .5 * ((A[23] - A[22]) + (B[1] - B[0]))
            e = (((B[0] - A[23]) - pg) / sc[v]) ** 2
            c += WT[v] * (50 if np.isnan(e) else e)
        return c

    def twin(d, pool):
        if not pool:
            return []
        dist = np.sqrt(np.nanmean((Z.loc[pool].values - Z.loc[d].values) ** 2, axis=1))
        return [p for p, o in zip(pool, dist <= .05) if o]

    # pass-1 dates (group exact twins with the previous record)
    groups = []
    for d in p1:
        if groups and twin(d, [groups[-1][-1]]):
            groups[-1].append(d)
        else:
            groups.append([d])
    rep = [g[0] for g in groups]
    n = len(rep)
    C = np.array([[cost(a, b) if a != b else np.inf for b in rep] for a in rep])
    order = list(range(n))
    tot = lambda o: sum(C[o[i], o[i + 1]] for i in range(len(o) - 1))
    best = tot(order); improved = True
    while improved:
        improved = False
        for L in (1, 2, 3):
            for i in range(n - L + 1):
                seg = order[i:i + L]; rest = order[:i] + order[i + L:]
                for j in range(max(0, i - 8), min(len(rest), i + 8) + 1):
                    if j == i:
                        continue
                    cand = rest[:j] + seg + rest[j:]
                    t = tot(cand)
                    if t < best - 1e-9:
                        order, best, improved = cand, t, True
                        break
                if improved:
                    break
            if improved:
                break
    links = np.array([C[order[i], order[i + 1]] for i in range(n - 1)])
    thr = np.quantile(links, .975)
    cal, c = {}, 0.0
    for k, gi in enumerate(order):
        if k > 0 and links[k - 1] > thr:
            c += GAP
        for d in groups[gi]:
            cal[(farm, d)] = c
        c += 1
    # pass 2, causal
    rl = roles[roles.farm == farm].set_index("day").role
    prev = None
    for d in [d for d in days if d >= 179]:
        earlier = [e for e in days if e < d]
        tw = twin(d, earlier)
        if tw:
            cal[(farm, d)] = float(np.mean([cal[(farm, e)] for e in tw]))
        else:
            cal[(farm, d)] = cal[(farm, prev)] + (0.0 if rl.get(d) == "second" else 1.0)
        prev = d
    return cal, dict(n_dates=n, moved=int(sum(o != i for i, o in enumerate(order))), gaps=int((links > thr).sum()))


def calendar():
    X, roles = load()
    out, info = {}, {}
    for f in ("F13", "F47"):
        c, i = build(X, roles, f)
        out.update(c); info[f] = i
    return out, info


if __name__ == "__main__":
    cal, info = calendar()
    print(info)
    C = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_11_days.csv"))[["farm", "day", "cal"]]
    C["own"] = [cal.get((f, d), np.nan) for f, d in zip(C.farm, C.day)]
    for f in ("F13", "F47"):
        g = C[C.farm == f]
        for nm, m in (("pass-1", g.day < 179), ("pass-2", g.day >= 179)):
            h = g[m]
            print(f, nm, "Spearman own vs deep %.3f" % h[["own", "cal"]].corr("spearman").iloc[0, 1])
        # neighbour agreement: for each pass-1 date pair adjacent in deep cal, adjacent in own?
        h = g[g.day < 179].drop_duplicates("cal").sort_values("cal")
        own_rank = h.own.rank(method="dense")
        print(f, "pass-1 consecutive deep dates that are consecutive in own: %.2f" % (np.abs(np.diff(own_rank.values)) == 1).mean())
    C.to_csv(os.path.join(env.LOCAL, "cal_own_farm_v1.csv"), index=False)
