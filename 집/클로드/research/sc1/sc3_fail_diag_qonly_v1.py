# -*- coding: utf-8 -*-
"""SC3 (2026-10-12 집 클로드; 사용자: "fail 이유 더 찾아보고 성능은 유지한 채로 개선할 수 있는지").  탐색(EV·DIAG10 은 이미 본 검증기 — 확정 아님).
진단: 기록별 BASE 계절 vs 같은 날짜 1차 계절(Δs), 그리고 팔별 기록 ΔSSE 를 |Δs| 구간별로 집계.
팔: BASE / BOTH(=SC2 P2DATE: 학습·검증 2차 계절 모두 교체) / QONLY(학습 계절은 BASE 그대로, 검증 기록 계절만 교체 — 모델은 BASE 와 동일)
    / QONLY5(QONLY 와 같되 |Δs| >= 5 인 검증 기록만 교체).  시드 47 / 1414 / 6464.
"""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import env_extra  # noqa: F401  (torch/tabpfn needed by model.py import)
import importlib.util, os, time
import numpy as np, pandas as pd
from pathlib import Path

PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
sys.path.insert(0, PK)
import model as M
from season import mapping, vectors, W
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor

OUT = Path(r"C:\work\farmai\집\클로드\research\sc1")
SEEDS = (7, 101, 2024)
THR, HMIN = .05, 5
tr = pd.read_pickle(r"C:\work\farmai\집\클로드\research\audit_ec\tr.pkl")
tr["day"] = tr.row_id.str[4:7].astype(int)
x = pd.read_csv(os.path.join(PK, "data", "train_X.csv"), usecols=["row_id"] + W)
VEC = vectors(x)
ALL = sorted(set(zip(tr.farm, tr.day.astype(int))))

P3 = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "ec_restart_phase3_20261001_v1")
sys.path.insert(0, P3)
spec = importlib.util.spec_from_file_location("p3_readonly", os.path.join(P3, "run_benchmark.py"))
p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)
_, _, _, LOCK, _, FDS = p3.prepare()


def common_mapping(tdays, qdays, vec):
    """season.mapping with F47 pass-1 records moved onto the F13 calendar (training weather only)."""
    from season import pav
    p1 = [(f, int(d)) for f, d in tdays.itertuples(index=False, name=None) if d < 179]
    f13 = [k for k in p1 if k[0] == "F13"]; f47 = sorted(k for k in p1 if k[0] == "F47")
    A = np.stack([vec[k].ravel() for k in f13]); offs = {}
    for k in f47:
        eq = (np.abs(A - vec[k].ravel()) < 1e-6).mean(1); j = int(np.argmax(eq))
        if eq[j] >= .9: offs[k[1]] = f13[j][1] - k[1]
    o = pd.Series(offs).sort_index(); o = o.rolling(7, center=True, min_periods=1).median()
    off = lambda d: float(np.interp(d, o.index.values, o.values))
    cs = lambda k: float(k[1]) if k[0] == "F13" else k[1] + off(k[1])
    reference = np.stack([vec[k] for k in p1])
    mu = np.nanmean(reference, axis=(0, 2)); sd = np.nanstd(reference, axis=(0, 2)); sd[sd == 0] = 1
    ref = (reference - mu[None, :, None]) / sd[None, :, None]
    seasons = {k: cs(k) for k in p1}; fits = {}
    for farm in ["F13", "F47"]:
        days = sorted(int(d) for f, d in tdays.itertuples(index=False, name=None) if f == farm and d >= 179)
        x, y, nearest = [], [], []
        for d in days:
            q = (vec[(farm, d)] - mu[:, None]) / sd[:, None]
            dist = np.sqrt(np.nanmean((ref - q) ** 2, axis=(1, 2)))
            nearest.append(cs(p1[int(np.nanargmin(dist))])); sel = np.flatnonzero(dist <= .05)
            if len(sel): x.append(d); y.append(float(np.mean([cs(p1[i]) for i in sel])))
        if len(x) < 2: x, y = days, nearest
        fitted = pav(y); x = np.asarray(x, float); fits[farm] = (x, fitted)
        for d in days: seasons[(farm, d)] = float(np.interp(d, x, fitted))
    out = []
    for farm, d in qdays.itertuples(index=False, name=None):
        if d < 179: out.append(float(d) if farm == "F13" else d + off(d))
        else: out.append(float(np.interp(d, *fits[farm])))
    return seasons, np.asarray(out), {"standardization": {"mu": mu.tolist(), "sd": sd.tolist()}, "n_offsets": len(offs)}


def folds():
    out = []
    for name, i, vd in FDS:
        if name == "DIAG10":
            out.append(("DIAG10", i, set(vd), set(LOCK)))
    fold = ((tr.day // 10) * 7 + (tr.farm == "F47") * 3) % 5
    for k in range(5):
        out.append(("BLK10", k, set(zip(tr.farm[fold == k], tr.day[fold == k].astype(int))), set()))
    ch = {}
    for f in ("F13", "F47"):
        days = sorted(d for ff, d in ALL if ff == f and d >= 179); runs = []; cur = [days[0]]
        for d in days[1:]:
            if d == cur[-1] + 1: cur.append(d)
            else: runs.append(cur); cur = [d]
        runs.append(cur); c = []
        for r in runs:
            for s in range(0, len(r), 5): c.append(r[s:s + 5])
        ch[f] = c
    assert len(ch["F13"]) == len(ch["F47"]) == 7, ch
    for j in range(7):
        out.append(("HOLE5", j, {("F13", d) for d in ch["F13"][j]} | {("F47", d) for d in ch["F47"][j]}, set()))
    return out


def twin_seasons(vrecs, refs, rseason, mu, sd, full=False):
    """vrecs: list of (farm, day) pass-2 validation records -> dict (f,d) -> array(24) season or nan (no match)."""
    R = np.stack([(VEC[k] - mu[:, None]) / sd[:, None] for k in refs]); rs = np.asarray(rseason, float); out = {}
    for k in vrecs:
        qz = (VEC[k] - mu[:, None]) / sd[:, None]; s = np.full(24, np.nan)
        if full:
            dist = np.sqrt(np.nanmean((R - qz[None]) ** 2, axis=(1, 2))); sel = dist <= THR
            if sel.any(): s[:] = rs[sel].mean()
        else:
            for h in range(HMIN, 24):
                dist = np.sqrt(np.nanmean((R[:, :, :h + 1] - qz[None, :, :h + 1]) ** 2, axis=(1, 2))); sel = dist <= THR
                if sel.any(): s[h] = rs[sel].mean()
        out[k] = s
    return out


def fit(trn, seed):
    e = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=250, max_features=1., n_jobs=8, random_state=seed))
    e.fit(trn[M.FULL_R3], trn.sub_ec); l = M.lg(seed); l.set_params(n_jobs=8); l.fit(trn[M.BASE_R3], trn.sub_ec)
    return e, l


def pred(m, frame, lo, hi):
    e, l = m
    p = (.6 * e.predict(frame[M.FULL_R3]) + .3 * l.predict(frame[M.BASE_R3])) / .9
    return np.clip(M.shrink(p, frame), lo, hi)



SEEDS = (47, 1414, 6464)
FV = pd.read_csv(r"C:\work\farmai\집\클로드\research\ev_validator\ev_folds_v1.csv")


def folds_ev():
    out = []
    for name, i, vd, lock in folds():
        if name == "DIAG10": out.append((name, i, vd, lock, set()))
    for k in sorted(FV.fold.unique()):
        G = FV[FV.fold == k]
        out.append(("EV", k, set(zip(G[G.role == "val"].farm, G[G.role == "val"].num.astype(int))), set(), set(zip(G[G.role == "del"].farm, G[G.role == "del"].num.astype(int)))))
    return out


def exact_season(ts, keys, avail):
    p1 = [k for k in ts if k[1] < 179 and k in avail]
    A = {f: [k for k in p1 if k[0] == f] for f in ("F13", "F47")}
    M = {f: np.stack([VEC[k].ravel() for k in A[f]]) for f in A}
    out = {}
    for k in keys:
        if k[1] < 179: continue
        eq = (np.abs(M[k[0]] - VEC[k].ravel()) < 1e-6).mean(1); j = np.where(eq >= .9)[0]
        if len(j): out[k] = float(np.median([ts[A[k[0]][x]] for x in j]))
    return out


def main():
    t0 = time.time(); rows = []; srec = []
    for name, i, vd, lock, dele in folds_ev():
        vrec = sorted(k for k in vd if k in set(ALL))
        if not any(k[1] >= 179 for k in vrec): continue
        forb = ({(f, d + j) for f, d in vd | lock for j in (-1, 0, 1)} if name == "DIAG10" else set(vd) | set(dele))
        key = list(zip(tr.farm, tr.day.astype(int)))
        trm = np.array([k not in forb for k in key]); vam = np.array([k in set(vrec) for k in key])
        tdays = tr[trm][["farm", "day"]].drop_duplicates(); qd = pd.DataFrame(vrec, columns=["farm", "day"])
        ts, qs, notes = mapping(tdays, qd, VEC); qs = np.asarray(qs, float)
        avail = set(zip(tdays.farm, tdays.day.astype(int)))
        ex_t = exact_season(ts, list(ts), avail); ex_q = exact_season(ts, vrec, avail)
        ts_both = dict(ts); ts_both.update(ex_t)
        q_ex = np.array([ex_q.get(k, qs[t]) for t, k in enumerate(vrec)])
        q_ex5 = np.array([ex_q[k] if (k in ex_q and abs(ex_q[k] - qs[t]) >= 5) else qs[t] for t, k in enumerate(vrec)])
        for t, k in enumerate(vrec):
            if k[1] >= 179: srec.append(dict(validator=name, fold=i, rec="%s_%03d" % k, base=qs[t], exact=ex_q.get(k, np.nan)))
        trn0 = tr[trm].copy(); va = tr[vam].copy()
        trn0["season"] = [ts[(f, int(d))] for f, d in zip(trn0.farm, trn0.day)]
        trnB = tr[trm].copy(); trnB["season"] = [ts_both[(f, int(d))] for f, d in zip(trnB.farm, trnB.day)]
        lo, hi = trn0.sub_ec.min(), trn0.sub_ec.max()
        qmapper = lambda Q: [dict(zip(vrec, Q))[(f, int(d))] for f, d in zip(va.farm, va.day)]
        for seed in SEEDS:
            m0 = fit(trn0, seed); mB = fit(trnB, seed)
            for arm, m, Q in (("BASE", m0, qs), ("QONLY", m0, q_ex), ("QONLY5", m0, q_ex5), ("BOTH", mB, q_ex)):
                vv = va.copy(); vv["season"] = qmapper(Q)
                rows.append(pd.DataFrame(dict(validator=name, fold=i, seed=seed, arm=arm, row_id=vv.row_id.values, farm=vv.farm.values,
                                              day=vv.day.values, y=vv.sub_ec.values, p=pred(m, vv, lo, hi))))
        print("%s fold %s | %.0fs" % (name, i, time.time() - t0), flush=True)
    O = pd.concat(rows, ignore_index=True); O.to_csv(OUT / "sc3_cv_oof.csv", index=False)
    S = pd.DataFrame(srec); S["ds"] = S.exact - S.base; S.to_csv(OUT / "sc3_seasons.csv", index=False)
    print("\nvalidation pass-2 records: |season correction| distribution", S.groupby("validator").ds.apply(lambda x: np.round(np.nanquantile(np.abs(x), [.5, .75, .9, 1]), 1).tolist()).to_dict())
    print("largest corrections (EV):"); print(S[S.validator == "EV"].reindex(S[S.validator == "EV"].ds.abs().sort_values(ascending=False).index).head(10).round(1).to_string(index=False))
    O["rec"] = O.row_id.str[:7]; r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    ch = dict(zip(FV[FV.role == "val"].rec, FV[FV.role == "val"].chunk)); rng = np.random.default_rng(20261235); res = []; per_all = []
    for (v, sd_), G in O[O.day >= 179].groupby(["validator", "seed"]):
        B = G[G.arm == "BASE"].set_index("row_id")
        for arm in ("QONLY", "QONLY5", "BOTH"):
            A_ = G[G.arm == arm].set_index("row_id").reindex(B.index)
            per = ((A_.p - A_.y) ** 2 - (B.p - B.y) ** 2).groupby(B.rec).sum()
            grp = per.groupby([ch.get(x, x) for x in per.index]).sum() if v == "EV" else per
            ii = rng.integers(0, len(grp), (4000, len(grp))); pw = float((grp.values[ii].sum(1) >= 0).mean())
            top = float(per.min() / per.sum()) if per.sum() < 0 else np.nan
            res.append(dict(validator=v, seed=sd_, arm=arm, pass2=100 * (r(A_.p - A_.y) / r(B.p - B.y) - 1), P_worse=pw, top_share=top, worse=int((per > 1e-9).sum()), better=int((per < -1e-9).sum())))
            for rc, dv in per.items(): per_all.append(dict(validator=v, seed=sd_, arm=arm, rec=rc, dsse=dv))
    R = pd.DataFrame(res); print(R.round(3).to_string(index=False)); R.to_csv(OUT / "sc3_summary.csv", index=False)
    P = pd.DataFrame(per_all).merge(S.groupby(["validator", "rec"]).ds.first().reset_index(), on=["validator", "rec"], how="left")
    P["band"] = pd.cut(P.ds.abs().fillna(0), [-.1, 1, 5, 10, 100], labels=["<=1", "1-5", "5-10", ">10"])
    print("\nmean dSSE per record by |season correction| band (mean over seeds) and share of records improved:")
    T_ = P.groupby(["validator", "arm", "band"]).agg(n=("dsse", "size"), mean_dsse=("dsse", "mean"), improved=("dsse", lambda x: float((x < -1e-9).mean())))
    T_["n"] = T_["n"] / 3; print(T_.round(3).to_string())
    P.to_csv(OUT / "sc3_records.csv", index=False)


if __name__ == "__main__":
    main()
