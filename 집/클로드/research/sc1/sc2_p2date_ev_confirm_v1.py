# -*- coding: utf-8 -*-
"""SC2 (2026-10-12 집 클로드; SC1/SC1b 뒤 새 검증기 확인 — 사전 고정, 실행 전 커밋).
SC1b 에서 일관된 몫 = '1차와 같은 날짜에 있는 2차 기록'(TRUE·NOLAB, 세 검증기 모두 이득); 순수 2차 창 몫은 검증기·팔마다 부호가 바뀜.
그래서 그 성분만 떼어 낸 팔을 처음 쓰는 검증기(EV, ev_validator/ev_folds_v1.csv: val 기록 숨기고 del 기록도 학습에서 제외)로 판정.
팔: BASE = 제출 계절표(mapping).  P2DATE = BASE 와 같되, 2차 기록(학습·검증 모두) 중 외기 24h 가 90% 이상 같은 '같은 농장 1차 학습 기록'
     (그 폴드에서 숨기지 않은 것)이 있으면 계절 값 = 그 1차 기록들의 BASE 계절 값 중앙(= 같은 날짜 1차 번호), 없으면 BASE 그대로.
     정답·확실 연결 사용 안 함(외기 입력 + 학습 기록만).
대리 모델: SF1 과 같음(ET .6 + LGB .3).  시드: 47 / 1414 / 6464 (처음 쓰는 시드).
판정(EV 만): 2차 검증 행 RMSE 가 세 시드 모두 BASE 보다 좋음 AND EV 덩어리(chunk) 부트스트랩 P(worse) < .025 (시드마다) AND 최대 기록 몫 <= 50%.
DIAG10 은 발견에 쓴 검증기라 참고로만 보고.
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


def p2date(ts, qkeys, qs, avail):
    p1 = [k for k in ts if k[1] < 179 and k in avail]
    A = {f: [k for k in p1 if k[0] == f] for f in ("F13", "F47")}
    M = {f: np.stack([VEC[k].ravel() for k in A[f]]) for f in A}
    def same(k):
        eq = (np.abs(M[k[0]] - VEC[k].ravel()) < 1e-6).mean(1); j = np.where(eq >= .9)[0]
        return [A[k[0]][x] for x in j]
    ts2 = dict(ts); n_tr = 0
    for k in ts:
        if k[1] >= 179:
            s = same(k)
            if s: ts2[k] = float(np.median([ts[x] for x in s])); n_tr += 1
    qs2 = qs.copy(); n_q = 0
    for t, k in enumerate(qkeys):
        if k[1] >= 179:
            s = same(k)
            if s: qs2[t] = float(np.median([ts[x] for x in s])); n_q += 1
    return ts2, qs2, n_tr, n_q


def main():
    t0 = time.time(); rows = []; cov = []
    for name, i, vd, lock, dele in folds_ev():
        vrec = sorted(k for k in vd if k in set(ALL))
        if not any(k[1] >= 179 for k in vrec): continue
        forb = ({(f, d + j) for f, d in vd | lock for j in (-1, 0, 1)} if name == "DIAG10" else set(vd) | set(dele))
        key = list(zip(tr.farm, tr.day.astype(int)))
        trm = np.array([k not in forb for k in key]); vam = np.array([k in set(vrec) for k in key])
        tdays = tr[trm][["farm", "day"]].drop_duplicates(); qd = pd.DataFrame(vrec, columns=["farm", "day"])
        ts, qs, notes = mapping(tdays, qd, VEC)
        ts2, qs2, n_tr, n_q = p2date(ts, vrec, np.asarray(qs, float), set(zip(tdays.farm, tdays.day.astype(int))))
        cov.append((name, i, n_tr, n_q, sum(1 for k in vrec if k[1] >= 179)))
        for arm, (TS, QS) in (("BASE", (ts, qs)), ("P2DATE", (ts2, qs2))):
            trn = tr[trm].copy(); va = tr[vam].copy()
            trn["season"] = [TS[(f, int(d))] for f, d in zip(trn.farm, trn.day)]
            qmap = dict(zip(vrec, QS)); va["season"] = [qmap[(f, int(d))] for f, d in zip(va.farm, va.day)]
            lo, hi = trn.sub_ec.min(), trn.sub_ec.max()
            for seed in SEEDS:
                m = fit(trn, seed)
                rows.append(pd.DataFrame(dict(validator=name, fold=i, seed=seed, arm=arm, row_id=va.row_id.values, farm=va.farm.values,
                                              day=va.day.values, hour=va.hour.values, y=va.sub_ec.values, p=pred(m, va, lo, hi))))
        print("%s fold %s: val %d | pass-2 seasons changed: train %d, val %d of %d | %.0fs" % (name, i, len(vrec), n_tr, n_q, cov[-1][4], time.time() - t0), flush=True)
    O = pd.concat(rows, ignore_index=True); O.to_csv(OUT / "sc2_cv_oof.csv", index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261234); res = []
    ch = dict(zip(FV[FV.role == "val"].rec, FV[FV.role == "val"].chunk))
    for (v, sd_), G in O.groupby(["validator", "seed"]):
        B = G[G.arm == "BASE"].set_index("row_id"); S_ = G[G.arm == "P2DATE"].set_index("row_id").reindex(B.index); p2 = B.day >= 179
        d2 = r((S_.p - S_.y)[p2]) / r((B.p - B.y)[p2]) - 1; da = r(S_.p - S_.y) / r(B.p - B.y) - 1
        rec = B.farm[p2] + "_" + B.day[p2].astype(int).astype(str).str.zfill(3)
        per = ((S_.p - S_.y) ** 2 - (B.p - B.y) ** 2)[p2].groupby(rec).sum()
        grp = per.groupby([ch.get(x, x) for x in per.index]).sum() if v == "EV" else per
        idx = rng.integers(0, len(grp), (4000, len(grp))); pw = float((grp.values[idx].sum(1) >= 0).mean())
        top = float(per.min() / per.sum()) if per.sum() < 0 else np.nan
        res.append(dict(validator=v, seed=sd_, pass2=100 * d2, all=100 * da, P_worse=pw, top_share=top, changed=int((per != 0).sum()), n=len(per)))
        if v == "EV": print("  EV seed", sd_, "largest:", ", ".join("%s %+.2f" % (a, b) for a, b in list(per.sort_values().items())[:4] + list(per.sort_values().items())[-3:]))
    R = pd.DataFrame(res); print(R.round(3).to_string(index=False)); R.to_csv(OUT / "sc2_cv_summary.csv", index=False)
    E = R[R.validator == "EV"]
    ok = (E.pass2 < 0).all() and (E.P_worse < .025).all() and (E.top_share <= .5).all()
    print("VERDICT SC2 (EV):", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
