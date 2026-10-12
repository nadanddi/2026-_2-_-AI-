# -*- coding: utf-8 -*-
"""SC1b (2026-10-12 집 클로드, 사후 진단 — 판정 변경용 아님): SC1 의 큰 이득이 '정답 연결로 만든 달력'의 누수인지,
어느 날짜 종류에서 오는지.  팔: BASE(제출 계절표), TRUE(SC1 과 같음), NOLAB(정답 미사용 달력: 외기 묶음의 1차 F13 번호 중앙값 순위로
1차 날짜 순번 매김, 1차 F13 이 없는 묶음과 순수 2차 창은 같은 농장·차수 안에서 번호로 보간 — 확실 연결·정답 전혀 사용 안 함).
보고: 팔별 9칸 변화·P, 그리고 TRUE/NOLAB 의 2차 기록별 ΔSSE 를 날짜 종류(1차 날짜 / 순수 2차 창 / 끼어든 날짜 / 미확정)별 합.
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



CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv")
CAL["p2"] = CAL.num >= 179
KIND = {(f, int(n)): k for f, n, k in zip(CAL.farm, CAL.num, CAL.kind)}
for (f, p2), g in CAL.groupby(["farm", "p2"]):
    ok = g.day.notna(); CAL.loc[g.index[~ok], "day"] = np.interp(g.num[~ok], g.num[ok], g.day[ok])
TRUEDAY = {(f, int(n)): float(d) for f, n, d in zip(CAL.farm, CAL.num, CAL.day)}
# NOLAB calendar: outdoor groups, rank by F13 pass-1 median number, others interpolated by number within farm/pass
_keys = sorted(VEC)
_A = np.stack([VEC[k].ravel() for k in _keys]); _par = list(range(len(_keys)))
def _fd(x):
    while _par[x] != x: _par[x] = _par[_par[x]]; x = _par[x]
    return x
for _i in range(len(_keys)):
    _eq = (np.abs(_A[_i + 1:] - _A[_i]) < 1e-6).mean(1)
    for _j in np.where(_eq >= .9)[0] + _i + 1: _par[_fd(_j)] = _fd(_i)
_grp = {}
for _i, k in enumerate(_keys): _grp.setdefault(_fd(_i), []).append(k)
_pos = {c: np.median([d for f, d in v if f == "F13" and d < 179]) for c, v in _grp.items() if any(f == "F13" and d < 179 for f, d in v)}
_rank = {c: r + 1 for r, c in enumerate(sorted(_pos, key=lambda c: _pos[c]))}
NL = pd.DataFrame([dict(farm=k[0], num=k[1], day=_rank.get(_fd(i), np.nan)) for i, k in enumerate(_keys)]); NL["p2"] = NL.num >= 179
for (f, p2), g in NL.groupby(["farm", "p2"]):
    ok = g.day.notna(); NL.loc[g.index[~ok], "day"] = np.interp(g.num[~ok], g.num[ok], g.day[ok])
NOLAB = {(f, int(n)): float(d) for f, n, d in zip(NL.farm, NL.num, NL.day)}


def dict_mapping(D):
    def mp(tdays, qdays, vec):
        return ({(f, int(d)): D[(f, int(d))] for f, d in tdays.itertuples(index=False, name=None)},
                np.array([D[(f, int(d))] for f, d in qdays.itertuples(index=False, name=None)]), {})
    return mp


def main():
    t0 = time.time(); rows = []
    for name, i, vd, lock in folds():
        vrec = sorted(k for k in vd if k in set(ALL))
        if not any(k[1] >= 179 for k in vrec): continue
        forb = {(f, d + j) for f, d in vd | lock for j in (-1, 0, 1)}
        key = list(zip(tr.farm, tr.day.astype(int)))
        trm = np.array([k not in forb for k in key]); vam = np.array([k in set(vrec) for k in key])
        tdays = tr[trm][["farm", "day"]].drop_duplicates(); qd = pd.DataFrame(vrec, columns=["farm", "day"])
        for arm, mp in (("BASE", mapping), ("TRUE", dict_mapping(TRUEDAY)), ("NOLAB", dict_mapping(NOLAB))):
            ts, qs, notes = mp(tdays, qd, VEC)
            trn = tr[trm].copy(); va = tr[vam].copy()
            trn["season"] = [ts[(f, int(d))] for f, d in zip(trn.farm, trn.day)]
            qmap = dict(zip(vrec, qs)); va["season"] = [qmap[(f, int(d))] for f, d in zip(va.farm, va.day)]
            lo, hi = trn.sub_ec.min(), trn.sub_ec.max()
            for seed in SEEDS:
                m = fit(trn, seed)
                rows.append(pd.DataFrame(dict(validator=name, fold=i, seed=seed, arm=arm, row_id=va.row_id.values, farm=va.farm.values,
                                              day=va.day.values, hour=va.hour.values, y=va.sub_ec.values, p=pred(m, va, lo, hi))))
        print("%s fold %s | %.0fs" % (name, i, time.time() - t0), flush=True)
    O = pd.concat(rows, ignore_index=True); O.to_csv(OUT / "sc1b_cv_oof.csv", index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261233); res = []; kinds = []
    for (v, sd_), G in O.groupby(["validator", "seed"]):
        B = G[G.arm == "BASE"].set_index("row_id"); p2 = B.day >= 179
        for arm in ("TRUE", "NOLAB"):
            S_ = G[G.arm == arm].set_index("row_id").reindex(B.index)
            d2 = r((S_.p - S_.y)[p2]) / r((B.p - B.y)[p2]) - 1; da = r(S_.p - S_.y) / r(B.p - B.y) - 1
            per = ((S_.p - S_.y) ** 2 - (B.p - B.y) ** 2)[p2].groupby(B.farm[p2] + "_" + B.day[p2].astype(int).astype(str).str.zfill(3)).sum()
            idx = rng.integers(0, len(per), (4000, len(per))); pw = float((per.values[idx].sum(1) >= 0).mean())
            res.append(dict(arm=arm, validator=v, seed=sd_, pass2=100 * d2, all=100 * da, P_worse=pw))
            for rec, dv in per.items():
                kinds.append(dict(arm=arm, validator=v, seed=sd_, rec=rec, kind=KIND.get((rec[:3], int(rec[4:])), "?"), dsse=dv))
    R = pd.DataFrame(res); print(R.round(3).to_string(index=False)); R.to_csv(OUT / "sc1b_cv_summary.csv", index=False)
    K = pd.DataFrame(kinds); K.to_csv(OUT / "sc1b_records.csv", index=False)
    print("\npass-2 dSSE by date kind (sum over records, mean over seeds):")
    print(K.groupby(["arm", "validator", "kind", "seed"]).dsse.sum().groupby(["arm", "validator", "kind"]).mean().round(2).to_string())
    print("\nlargest record gains/losses (TRUE, DIAG10, mean over seeds):")
    T_ = K[(K.arm == "TRUE") & (K.validator == "DIAG10")].groupby(["rec", "kind"]).dsse.mean().sort_values()
    print(T_.head(8).round(2).to_string()); print(T_.tail(5).round(2).to_string())


if __name__ == "__main__":
    main()
