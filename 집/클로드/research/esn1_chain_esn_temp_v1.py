# -*- coding: utf-8 -*-
"""ESN1 (2026-10-13 집 클로드; 사용자: 'Echo State Network 도 해봐').  사전 고정(커밋 후 실행).  온도.
CR1(같은 동 사슬 GRU, 6.803)과 같은 시퀀스·학습원(DIAG10 OOF 공통 + 이웃 제외)·판정, 모델만 ESN: 고정 무작위 저수지 200개(희소 10%, 스펙트럼 반경 .9,
누설률 .3, 입력 크기 .5, 시드 7/8) → 오늘 24단계의 저수지 상태 + 입력 → 릿지(λ 10) 선형 출력(학습되는 것은 출력층뿐).  모든 값 고정.
안: E2 = 앵커 전 셀 + ESN 보정 / E1 = 14회차형 앵커판 + ESN 보정.  판정은 CR1 원문 그대로(온도 범위 ALL, α .0125).
아래는 CR1 원문을 고친 것."""

import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s_ = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A7)
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
src = open(os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py"), encoding="utf-8").read()
_ns = {"__file__": os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py"), "__name__": "tb1src"}; exec(src[:src.index("PE15 = ")], _ns)
hp2, mirror_lab, ST, idx, TRAIN, FARM, NUM, N = (_ns[k] for k in ("hp2", "mirror_lab", "ST", "idx", "TRAIN", "FARM", "NUM", "N"))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261122); DEV = None
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")).set_index("rec")
PREV = {}; TNA8 = {}
for r in PM.index:
    pv, p0, lb = PM.at[r, "prev0"], PM.at[r, "p0"], PM.at[r, "lab0"]
    if isinstance(pv, str) and bool(lb) and (p0 >= .8 or hp2(idx[r], mirror_lab(r)) == pv):
        PREV[r] = pv
        if r not in v7set: TNA8[r] = pv
for r, pv in zip(L7f.rec, L7f.prev): PREV.setdefault(r, pv)
print(f"predecessors: {len(PREV)} records | TNA8 extra anchors (mirror): {len(TNA8)}", flush=True)
add = pd.DataFrame(dict(rec=list(TNA8), prev=list(TNA8.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNA8.values()], dT_air=0.0, steps_to_anchor=1))
LC = pd.concat([L7f, add], ignore_index=True)
V = ["in_temp", "in_hum", "out_temp", "out_rad", "act_heating", "act_thermal", "act_vent"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
CUBE = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in V], 2)   # [N,24,7]
def build(recs, qmap, hidden):
    """recs: list of record ids; qmap: rec -> base array[24]; hidden: set of hidden recs (labels not visible)"""
    Xs = np.zeros((len(recs), 48, len(V) + 1 + 3 + 2), np.float32)
    for k, r in enumerate(recs):
        b = idx[r]; pv = PREV.get(r)
        if pv is not None:
            a = idx[pv]; Xs[k, :24, :len(V)] = CUBE[a]
            vis = TRAIN[a] and pv not in hidden
            if vis: Xs[k, :24, len(V)] = ST[a]
            Xs[k, :24, len(V) + 1] = 1.0; Xs[k, :24, len(V) + 2] = float(vis)
        Xs[k, 24:, :len(V)] = CUBE[b]; Xs[k, 24:, len(V) + 3] = qmap[r]; Xs[k, 24:, len(V) + 4] = np.arange(24) / 23
    return Xs
RS = np.random.default_rng(7); NRES = 200; LEAK, RHO, INS, LAM = .3, .9, .5, 10.0
_W = RS.standard_normal((NRES, NRES)) * (RS.random((NRES, NRES)) < .1); _W *= RHO / max(abs(np.linalg.eigvals(_W)))
_WIN = None
def states(Z):
    global _WIN
    if _WIN is None: _WIN = np.random.default_rng(8).uniform(-INS, INS, (NRES, Z.shape[2] + 1))
    S = np.zeros((Z.shape[0], 24, NRES)); x = np.zeros((Z.shape[0], NRES))
    for t in range(48):
        u = np.c_[Z[:, t, :], np.ones(Z.shape[0])]; x = (1 - LEAK) * x + LEAK * np.tanh(u @ _WIN.T + x @ _W.T)
        if t >= 24: S[:, t - 24] = x
    return S
def fit_predict(Xtr, Ytr, Xte):
    mu = np.nanmean(Xtr.reshape(-1, Xtr.shape[2]), 0); sd = np.nanstd(Xtr.reshape(-1, Xtr.shape[2]), 0) + 1e-6
    for flag in (len(V) + 1, len(V) + 2): mu[flag] = 0; sd[flag] = 1
    z = lambda A: np.nan_to_num((A - mu) / sd)
    Ztr, Zte = z(Xtr), z(Xte); Str, Ste = states(Ztr), states(Zte)
    Ftr = np.concatenate([Str, Ztr[:, 24:, :]], 2).reshape(-1, NRES + Ztr.shape[2]); Fte = np.concatenate([Ste, Zte[:, 24:, :]], 2).reshape(-1, NRES + Zte.shape[2])
    y = Ytr.reshape(-1); m = np.isfinite(y); Ft = np.c_[Ftr[m], np.ones(m.sum())]
    w = np.linalg.solve(Ft.T @ Ft + LAM * np.eye(Ft.shape[1]), Ft.T @ y[m])
    return (np.c_[Fte, np.ones(len(Fte))] @ w).reshape(len(Xte), 24)
res = []; blocks = {}
FR = A7.A2.A1.frames()
def layout_maps(M, cells):
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values; y = M.sub_temp.values
    kb, Ab = A7.build(M, LC); anc = [A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5) for c in cells]
    qbar = np.mean(cells, 0); abar = np.mean(anc, 0); recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    def per_rec(arr): return {r: np.array([arr[pos[r]][h[pos[r]] == t][0] if (h[pos[r]] == t).any() else np.nan for t in range(24)]) for r in recs}
    return key, h, y, anc, recs, pos, per_rec(qbar), per_rec(abar), per_rec(y)
_, _, _, _, DREC, _, DQ, DA, DY = layout_maps(*FR["DIAG10"])   # common training source (DIAG10 OOF), purged per test fold
def purge(te):
    fd = {(FARM[idx[r]], int(NUM[idx[r]])) for r in te}; out = set()
    for r in DREC:
        f, d = FARM[idx[r]], int(NUM[idx[r]])
        if any((f == f2 and abs(d - d2) <= 1) or (f != f2 and abs(d - d2) <= 5) for f2, d2 in fd): out.add(r)
    return out
for L, (M, cells) in FR.items():
    key, h, y, anc, recs, pos, Q, A, Y = layout_maps(M, cells)
    fold_of = {r: M.fold.values[pos[r][0]] for r in recs}; corr2 = {}; corr1 = {}
    for k in sorted(set(fold_of.values())):
        te = [r for r in recs if fold_of[r] == k]; hid = set(te); tr = [r for r in DREC if r not in purge(te)]
        for (base, dbase), store in (((Q, DQ), corr2), ((A, DA), corr1)):
            Xtr = build(tr, dbase, hid); Ytr = np.stack([DY[r] - dbase[r] for r in tr]); Xte = build(te, base, hid)
            P = fit_predict(Xtr, Ytr, Xte)
            for r, p in zip(te, P): store[r] = p
    def apply(arrs, corr):
        outs = []
        for c in arrs:
            n = c.copy()
            for r in recs:
                ix = pos[r]; n[ix] = c[ix] + corr[r][h[ix]]
            outs.append(n)
        return outs
    g2 = apply(cells, corr2); g1 = apply(anc, corr1)
    for ci in range(len(cells)):
        for a, new in (("E2", g2[ci]), ("E1", g1[ci])):
            base = anc[ci]; p2 = M.day.values >= 179
            res.append(dict(layout=L, cell=ci, arm=a, pct=100 * (rm(new - y) / rm(base - y) - 1), p2=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)))
            if ci == 0:
                bk = M.farm + "_" + (M.day // 5).astype(str); blocks[(L, a)] = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(bk.values).mean()
    print("done", L, flush=True)
S = pd.DataFrame(res); pd.set_option("display.width", 220)
print(S.pivot_table(index=["arm", "layout"], columns="cell", values="pct").round(3).to_string())
print(S.groupby(["arm", "layout"]).p2.mean().round(3).unstack(0).to_string())
def bootP(L, a):
    v = blocks[(L, a)].values; ix = rng.integers(0, len(v), (20000, len(v))); m = v[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for a in ("E2", "E1"):
    ok = True; msg = []
    for L in sorted(set(S.layout)):
        cells_ = S[(S.arm == a) & (S.layout == L)].pct; pw, pb = bootP(L, a)
        if L in ("DIAG10", "EXT10"):
            c = bool((cells_ < 0).all()) and pw < .0125; ok &= c; msg.append(f"{L}[primary] cells {int((cells_ < 0).sum())}/4 P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}")
        else:
            bad = cells_.mean() > 0 and pb < .0125; ok &= not bad; msg.append(f"{L} mean {cells_.mean():+.3f}% P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}")
    print(f"\n== ESN1 {a}:"); print("\n".join("   " + m for m in msg)); print(f"ESN1 {a}:", "ADOPT" if ok else "FAIL")
