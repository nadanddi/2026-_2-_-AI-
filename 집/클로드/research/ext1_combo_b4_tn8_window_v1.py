# -*- coding: utf-8 -*-
"""EXT1 (2026-10-13 집 클로드; 사용자: '해볼 가치 있는 것 전부 + 더 뻗어나갈 방식').  사전 고정(커밋 후 실행).  온도.
(1) COMBO = B4(사슬 새벽 보정 전날 = v15R p0 ≥ .5 짝, 0~7시) 셀 + 13회차 앵커(v7, 측정 감쇠) + TN8 앵커(v15R p0 ≥ .8, 정답, v7 없음)
    vs 13회차 구성(TC1c 셀 + v7 앵커).  기준: DIAG10·EL1·F~K 모든 칸 개선 & 합친(DIAG10+F~K 평균) 5일 블록 P(악화) < .025 → 조합 확인.
(2) B4X = 사슬 보정 시간 창 넓히기(TC2 6.555 는 옛 사슬로 0~11시 +.2~.3% 악화) — v15R 사슬 셀에서 H 11·H 23 vs H 7, 앵커는 COMBO 와 같게.
    안 2개 α .0125: 모든 칸(DIAG10·EL1·F~K) 개선 & 합친 P < .0125 → ADOPT.  EXT10·EXT12 서술.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s_ = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A7)
A1 = A7.A2.A1; T, B, C2 = A1.T, A1.B, A1.C2
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261119)
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import ST, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
IN23 = _X[_X.h == 23].set_index("rec").in_temp; EVE = _X[_X.h >= 18].groupby("rec").in_temp.mean()
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")).set_index("rec")
CT = T.chain_table(); CT2 = CT.copy()
for i, r in CT2.iterrows():
    k = f"{r.farm}_{int(r.day):03d}"
    if k in PM.index and isinstance(PM.at[k, "prev0"], str) and PM.at[k, "p0"] >= .5 and PM.at[k, "prev0"] in IN23.index:
        CT2.at[i, "ch_in23"] = IN23[PM.at[k, "prev0"]]; CT2.at[i, "ch_eve"] = EVE[PM.at[k, "prev0"]]
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
TN8 = {r: pv for r, pv, p0, lb in zip(PM.index, PM.prev0, PM.p0, PM.lab0) if r not in v7set and bool(lb) and isinstance(pv, str) and p0 >= .8}
add = pd.DataFrame(dict(rec=list(TN8), prev=list(TN8.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TN8.values()], dT_air=0.0, steps_to_anchor=1))
LC = pd.concat([L7f, add], ignore_index=True)
print(f"TN8 mirror anchors {len(TN8)}", flush=True)
D1 = T.layout_frame("DIAG10", CT); D2 = T.layout_frame("DIAG10", CT2)
def cells(M, D, H, bs, cx):
    return [C2.corrected(M, B.w40(M, b, cx, fam), D, bi, fam, H) for bi, b in enumerate(bs) for fam in "AB"]
def anch(M, q, L):
    key, A = A7.build(M, L); return A7.correct(M, q, key, A, .8, "const", "meas", 2, 1.5)
res = {k: [] for k in ("COMBO", "B4X11", "B4X23")}; ok = {k: True for k in res}; diffs = {k: [] for k in res}
for L, (d, pre, bs, cx) in T.LAY.items():
    M1 = T.layout_frame(L, CT); M2 = T.layout_frame(L, CT2); y = M1.sub_temp.values
    base = [anch(M1, q, L7f) for q in cells(M1, D1, 7, bs, cx)]
    c7 = [anch(M2, q, LC) for q in cells(M2, D2, 7, bs, cx)]
    c11 = [anch(M2, q, LC) for q in cells(M2, D2, 11, bs, cx)]
    c23 = [anch(M2, q, LC) for q in cells(M2, D2, 23, bs, cx)]
    for k, (new, ref) in {"COMBO": (c7, base), "B4X11": (c11, c7), "B4X23": (c23, c7)}.items():
        ch = [rm(n - y) / rm(r_ - y) - 1 for n, r_ in zip(new, ref)]; p2 = M1.day.values >= 179
        ch2 = [rm((n - y)[p2]) / rm((r_ - y)[p2]) - 1 for n, r_ in zip(new, ref)]
        res[k].append(f"{L:6s} all {' '.join('%+.2f' % (100 * x) for x in ch)} | p2 {' '.join('%+.2f' % (100 * x) for x in ch2)}")
        if L not in ("EXT10", "EXT12"): ok[k] &= all(x < 0 for x in ch)
        if L not in ("EL1", "EXT10", "EXT12"):
            key = M1.farm + "_" + (M1.day // 5).astype(str); diffs[k].append(pd.Series((new[0] - y) ** 2 - (ref[0] - y) ** 2).groupby(key.values).mean())
    print("done", L, flush=True)
for k in res:
    S = pd.concat(diffs[k], axis=1).mean(axis=1).values; ix = rng.integers(0, len(S), (20000, len(S))); P = float((S[ix].mean(1) >= 0).mean())
    alpha = .025 if k == "COMBO" else .0125
    print(f"\n== {k}" + (" (vs 13th config)" if k == "COMBO" else " (vs B4+anchors at 0-7h)")); print("\n".join(res[k]))
    print(f"{k}: all cells better {ok[k]} | pooled P {P:.4f} -> {'PASS' if ok[k] and P < alpha else 'FAIL'}")
