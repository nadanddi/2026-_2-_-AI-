# -*- coding: utf-8 -*-
"""FL2 (2026-10-15 집 클로드; FL1 후속).  서술 비교(판정 아님), 실행 전 커밋.  링커 + 온도 + EC.
FL1 의 미리 정한 숨김(대상 ±2 모든 농장)이 진짜 전날까지 숨겨(cert 일치 .13~.14) 링커 비교가 성립하지 않았음 →
같은 비교를 '평가 상황과 같은 숨김'(v7 교차검증과 같은 거울 배치: 같은 농장 10번호 묶음 + 앞뒤 1, 다른 농장 ±2 이동 묶음)에서 다시.
거울 배치는 링커 가중치를 정하는 데 쓰지 않고(가중치 = truth 짝 전체 적합, FL1 과 같음) 검증 조건으로만 씀.
FULL(v7 특징 전부) vs FL(현장형 특징): 선택표 → 온도 v23 앵커(NONE/FULL/FL), EC F5E + EA 앵커(FULL/FL) — FL1 과 같은 지표.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u fl2_field_linker_mirror_eval_v1.py {tables|judge}
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); MODE = sys.argv[1] if len(sys.argv) > 1 else "tables"; sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]; IFL = [FEATS7.index(c) for c in FL]
def acc_cv(TRG, cols):
    blk = np.array([f"{FARM[b]}_{NUM[b] // 20}" for b, a, X_, y_ in TRG]); ub = sorted(set(blk)); fold = dict(zip(ub, np.random.default_rng(0).permutation(len(ub)) % 5)); ok = []
    for k in range(5):
        trg = [(X_[:, cols], y_) for (b, a, X_, y_), bb in zip(TRG, blk) if fold[bb] != k]; w = fit_clogit(trg, l2=0.5, iters=300)
        for (b, a, X_, y_), bb in zip(TRG, blk):
            if fold[bb] == k: s = X_[:, cols] @ w; ok.append(int(np.argmax(s) == y_))
    return float(np.mean(ok)), len(ok)
def pick_cols(b, lab, removed, w, cols):
    df, _ = features7(b, 0, lab, removed=removed)
    if len(df) == 0: return None, 0.0
    s = df[[FEATS7[i] for i in cols]].values.astype(float) @ w; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p)); return int(df.a.values[k]), float(p[k])
if MODE == "tables":
    TRG = train_groups7(); print("truth groups", len(TRG), flush=True)
    allc = list(range(len(FEATS7)))
    for nm, cols in (("FULL", allc), ("FL", IFL)):
        a, n = acc_cv(TRG, cols); print(f"(a) {nm}: top-1 accuracy on truth links (5-fold) {a:.3f} (n {n})", flush=True)
    IN_ = pd.read_pickle(os.path.join(S3, "base.pkl"))["IN"] if False else None
    for nm, cols in (("FULL", allc), ("FL", IFL)):
        w = fit_clogit([(X_[:, cols], y_) for b, a, X_, y_ in TRG], l2=0.5, iters=300); out = []
        for b in range(N):
            if not TRAIN[b] or FARM[b] not in ("F13", "F47"): continue
            f = FARM[b]; g_ = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min(); s0 = lo + ((NUM[b] - lo) // 10) * 10; m0 = s0 + sh
            touch = ((FARM == f) & (NUM >= s0 - 1) & (NUM <= s0 + 10)); oth = (FARM == g_) & (NUM >= m0 - 1) & (NUM <= m0 + 10); touch |= oth
            rem = set(np.where((FARM == f) & ((NUM == s0 - 1) | (NUM == s0 + 10)))[0].tolist()) | set(np.where(oth)[0].tolist())
            lab = TRAIN.copy(); lab[touch] = False
            a0, p0 = pick_cols(b, lab, rem, w, cols)
            if a0 is None: continue
            out.append(dict(rec=rid(b), prev=rid(a0), p=round(p0, 4), prev_is_labelled=bool(lab[a0]), xf=FARM[a0] != FARM[b], steps_to_anchor=1 if lab[a0] else -1,
                            anchor_st23=round(ST[a0, 23], 2) if lab[a0] else np.nan, anchor_ec23=round(EC[a0, 23], 3) if lab[a0] else np.nan))
        O = pd.DataFrame(out); O.to_csv(os.path.join(env.LOCAL, f"fl2_cv_{nm}.csv"), index=False, encoding="utf-8-sig")
        TLc = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(bb)): rid(int(aa)) for aa, bb, c in zip(TLc.a1, TLc.b, TLc.cert) if c}
        c = O[O.rec.isin(CERT.keys())]; print(f"(b) {nm}: table {len(O)} | acc vs cert {np.mean([CERT[r] == p for r, p in zip(c.rec, c.prev)]):.3f} (n {len(c)}) | p>=.8 labelled {int(((O.p >= .8) & O.prev_is_labelled).sum())}", flush=True)
    sys.exit(0)
# ---------------- judge ----------------
src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3fl2"}
exec(src[:src.index("JUDGE = [")], g)
T_, B_, C2_, anch_, D1, CT, rmf, IN23 = g["T"], g["B"], g["C2"], g["anch"], g["D1"], g["CT"], g["rm"], g["IN23"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
def lv(nm):
    O = pd.read_csv(os.path.join(env.LOCAL, f"fl2_cv_{nm}.csv"), encoding="utf-8-sig"); O["dT_air"] = [TIN.at[r, 0] - TIN.at[p, 23] if p in TIN.index else np.nan for r, p in zip(O.rec, O.prev)]
    O = O[(O.p >= .8) & O.prev_is_labelled.astype(bool) & (O.dT_air.abs() <= 1.5) & O.anchor_st23.notna()].copy(); O["p"] = 1.0; return O
LVS = {"FULL": lv("FULL"), "FL": lv("FL")}; EMPTY = LVS["FL"].iloc[:0]
print("anchors FULL", len(LVS["FULL"]), "FL", len(LVS["FL"]), "| same prev", len(set(zip(LVS["FULL"].rec, LVS["FULL"].prev)) & set(zip(LVS["FL"].rec, LVS["FL"].prev))), flush=True)
rng = np.random.default_rng(20270108); JUDGE = ["DIAG10", "EL1", "F", "G", "H", "I"]; ok = True; diffs = []; gain = {"FULL": [], "FL": []}
for L in JUDGE:
    d, pre, bs, cx = T_.LAY[L]; M = T_.layout_frame(L, CT); y = M.sub_temp.values; key = (M.farm + "_" + (M.day // 5).astype(str)).values; P = {k: [] for k in ("NONE", "FULL", "FL")}
    for bi, b in enumerate(bs):
        for fam in "AB":
            q = C2_.corrected(M, B_.w40(M, b, cx, fam), D1, bi, fam, 7); P["NONE"].append(anch_(M, q, EMPTY)); P["FULL"].append(anch_(M, q, LVS["FULL"])); P["FL"].append(anch_(M, q, LVS["FL"]))
    ch = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P["FL"], P["NONE"])]; chF = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P["FULL"], P["NONE"])]
    ok &= all(x < 0 for x in ch); gain["FL"].append(np.mean(ch)); gain["FULL"].append(np.mean(chF))
    if L != "EL1": diffs.append(pd.Series((P["FL"][0] - y) ** 2 - (P["NONE"][0] - y) ** 2).groupby(key).mean())
    print(f"[TEMP] {L:6s} FL vs NONE {' '.join('%+.2f' % (100 * x) for x in ch)} | FULL vs NONE {np.mean(chF) * 100:+.2f}", flush=True)
S = pd.concat(diffs, axis=1).mean(axis=1).values; Pt = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"[TEMP] FL anchors: all cells better {ok} | pooled P {Pt:.4f} -> {'PASS' if ok and Pt < .025 else 'FAIL'} | FL keeps {100 * np.mean(gain['FL']) / np.mean(gain['FULL']):.0f}% of FULL gain")
# EC
import pickle, glob
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC as ECA  # noqa: E402
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); MTp = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EP = pickle.load(open(os.path.join(env.LOCAL, "nh2_ev3l_v1.pkl"), "rb"))
mt = np.array([MTp[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); ev = np.array([EP[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
SEEDS = (1357, 2468, 3579); idx = {rid(i): i for i in range(N)}
for s in SEEDS: R[f"F5E_{s}"] = .9 * ((R[f"A1_{s}"] - .1 * mt) / .9) + .1 * ev
def ea(col, O):
    links = dict(zip(O.rec, O.prev)); out = R[col].values.copy()
    for (v, k, r), gg in R.groupby(["validator", "fold", "rec"]):
        p = links.get(r)
        if p is None or not np.isfinite(ECA[idx[p], 23]) or p in set(R.rec[(R.validator == v) & (R.fold == k)]): continue
        ii = gg.sort_values("hour").index.values; q = R.loc[ii, col].values; pm = np.cumsum(q) / np.arange(1, len(q) + 1); out[ii] = q + (ECA[idx[p], 23] - pm)
    return out
for nm in ("FULL", "FL"):
    for s in SEEDS: R[f"{nm}_{s}"] = ea(f"F5E_{s}", LVS[nm])
rows, blocks, recd = [], [], []
for v in ("DIAG10", "EL1", "P2LOO"):
    gg = R[R.validator == v]; y = gg.y.values; p2 = gg.p2.values; key = (gg.farm + "_" + (gg.day // 5).astype(str)).values
    for s in SEEDS:
        a, n, f_ = gg[f"F5E_{s}"].values, gg[f"FL_{s}"].values, gg[f"FULL_{s}"].values; dd = (n - y) ** 2 - (a - y) ** 2
        rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), dSSE_all=float(dd.sum()), FL_p2=100 * (rmf((n - y)[p2]) / rmf((a - y)[p2]) - 1), FULL_p2=100 * (rmf((f_ - y)[p2]) / rmf((a - y)[p2]) - 1)))
        if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
        for r_, x in pd.Series(dd[p2]).groupby(gg.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=x))
E = pd.DataFrame(rows); print(E.round(3).to_string(index=False))
main = E[E.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); x = pd.concat(blocks, axis=1).mean(axis=1).values; pP = float((x[rng.integers(0, len(x), (20000, len(x)))].mean(1) >= 0).mean())
RD = pd.DataFrame(recd); nh = int((RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c3 = not (E[E.val == "DIAG10"].dSSE_all.mean() > 0)
print(f"[EC] FL anchor vs F5E: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh} | DIAG10 mean dSSE {E[E.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .025 or nh == 0) and c3 else 'FAIL'}")
