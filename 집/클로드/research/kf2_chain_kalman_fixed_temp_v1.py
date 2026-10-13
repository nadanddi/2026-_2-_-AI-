# -*- coding: utf-8 -*-
"""KF2 (2026-10-13 집 클로드; 사용자: 'KF1 감사 결과대로 고쳐서 진행').  사전 고정(커밋 후 실행).  온도.
KF1(6.810) 감사(6.812)에서 확인한 결함 4개를 고침:
  (1) 이중 축소 제거: pred_h = q_h + (β_h/β_0)·clip(K·ĝ0, ±2)  (0시엔 칼만 K 그대로, 시간 흐름엔 측정 감쇠의 '모양'만).
  (2) 날짜 확인: 각 연결은 '짝의 학습 전용 달력(DMT3e) 날짜 = 대상 날짜 − 1 (같은 구간)' 이거나 (정답 짝일 때) 'HP2 짝 == v15R 짝' 이어야 함.
      날짜 = 0시 외기 4칸이 같은 보이는 정답 기록들의 DMT3e 날짜 최빈값(정답 기록은 자기 날짜).
  (3) 최대 2단계.   (4) 누적 짝 확신 P ≥ .5.
  나머지(상태 = 0시 차이, 정답 짝 분산 σ_L² .342, 정답 없는 단계 β23 .235·σ_h² .239, 사전 σ_s² .549, 편향 .36)는 KF1 측정값 그대로.
안(2개, α .0125): KF2a = 14회차형(v7 + TNA8 앵커) + 앵커 없는 기록에 KF2;  KF2b = KF2 를 낼 수 있는 기록은 앵커도 KF2 로 대체(나머지 14회차형).
판정(주 = DIAG10·EXT10·EV 세 하네스): 각각 4칸 모두 개선 & 농장×5일 블록 부트스트랩 P(악화) < .0125;  나머지 배치(EL1·EXT12·F~L)는 칸 평균 악화 & '후보가 나음' 비율 < .0125 면 FAIL.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py")
exec(src[:src.index("PE15 = ")])   # hp2, mirror_lab, v15_new, ST, idx, TRAIN, FARM, NUM, N, A7(BETA), rid, L7, v7set
from feat4 import X as OUTX  # noqa: E402
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261127)
S2, L2, H2, BIAS = .549, .342, .239, .36; B23 = float(A7.BETA[23]); SHAPE = A7.BETA / A7.BETA[0]
C3 = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY3 = {r: (d, s) for r, d, s in zip(C3.rec, C3.day, C3.seg)}
def date_of(r, lab):
    if lab[idx[r]] and r in DAY3: return DAY3[r]
    i = idx[r]; vi = np.where(lab)[0]; vi = vi[vi != i]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[i, 0]) < 1e-6, axis=1)]
    ds = [DAY3[rid(j)] for j in m if rid(j) in DAY3]
    return pd.Series(ds).mode().iloc[0] if ds else None
def kf2(r, picks, Qm, visible, inq, lab, hp2fn):
    chain = []; cur = r; P = 1.0
    for k in range(2):
        if cur not in picks: return None
        pv, p0 = picks[cur]
        if not isinstance(pv, str): return None
        P *= float(p0)
        if P < .5: return None
        vis = visible(pv) and np.isfinite(ST[idx[pv], 23])
        dc, dp = date_of(cur, lab), date_of(pv, lab)
        date_ok = dc is not None and dp is not None and dp[1] == dc[1] and dp[0] == dc[0] - 1
        if not date_ok and not (vis and hp2fn(cur) == pv): return None
        chain.append(pv)
        if vis: break
        if pv not in inq: return None
        cur = pv
    if not chain or not (visible(chain[-1]) and np.isfinite(ST[idx[chain[-1]], 23])): return None
    est23 = ST[idx[chain[-1]], 23]; v = L2
    for pv2 in reversed(chain[:-1]):
        g = (est23 - BIAS) - Qm[pv2][0]; est23 = Qm[pv2][23] + B23 * g; v = B23 ** 2 * v + H2
    g0 = (est23 - BIAS) - Qm[r][0]; return float(np.clip(S2 / (S2 + v) * P * g0, -2, 2)), len(chain)
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")); TNA8M = {**v15_new(PM, mirror_lab, v7set, 0.0, True), **v15_new(PM, mirror_lab, v7set, .8, False)}
add = pd.DataFrame(dict(rec=list(TNA8M), prev=list(TNA8M.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNA8M.values()], dT_air=0.0, steps_to_anchor=1))
LC = pd.concat([L7f, add], ignore_index=True); anchoredM = set(LC.rec)
pkM = {r: (pv, p0) for r, pv, p0 in zip(PM.rec, PM.prev0, PM.p0)}
res, blocks, fired_log = [], {}, []
def record(L, a, ci, M_day, y, base, new, farm):
    p2 = M_day >= 179
    res.append(dict(layout=L, cell=ci, arm=a, pct=100 * (rm(new - y) / rm(base - y) - 1), p2=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)))
    if ci == 0:
        bk = farm + "_" + (M_day // 5).astype(str); blocks[(L, a)] = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(np.asarray(bk)).mean()
for L, (M, cells) in A7.A2.A1.frames().items():
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values; y = M.sub_temp.values
    kb, Ab = A7.build(M, LC); anc = [A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5) for c in cells]
    recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}; qbar = np.mean(cells, 0)
    Qm = {r: np.array([qbar[pos[r]][h[pos[r]] == t][0] if (h[pos[r]] == t).any() else np.nan for t in range(24)]) for r in recs}
    fold_of = {r: M.fold.values[pos[r][0]] for r in recs}; hid = {}
    for r in recs: hid.setdefault(fold_of[r], set()).add(r)
    G = {}
    for r in recs:
        hs = hid[fold_of[r]]; lab = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
        out = kf2(r, pkM, Qm, lambda pv, hs=hs: bool(TRAIN[idx[pv]]) and pv not in hs, set(recs), lab, lambda cur: hp2(idx[cur], mirror_lab(cur)))
        if out: G[r] = out; fired_log.append(dict(layout=L, rec=r, steps=out[1], anchored=r in anchoredM))
    for ci, c in enumerate(cells):
        base = anc[ci]; a_, b_ = base.copy(), base.copy()
        for r, (g, k) in G.items():
            ix = pos[r]; v = c[ix] + SHAPE[h[ix]] * g
            if r not in anchoredM: a_[ix] = v
            b_[ix] = v
        record(L, "KF2a", ci, M.day.values, y, base, a_, M.farm.values); record(L, "KF2b", ci, M.day.values, y, base, b_, M.farm.values)
    print("done", L, flush=True)
# ---- EV harness ----
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidE = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
foldof = {r: k for k, g in FO[FO.role == "val"].groupby("fold") for r in g.rec}
lab_ev = lambda r: TRAIN & ~np.array([rid(i) in hidE[foldof[r]] for i in range(N)])
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv")); TNA8E = {**v15_new(PE, lab_ev, v7set, 0.0, True), **v15_new(PE, lab_ev, v7set, .8, False)}
pkE = {k: {r: (pv, p0) for r, pv, p0 in zip(g.rec, g.prev0, g.p0)} for k, g in PE.groupby("fold")}
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.reset_index(drop=True); y = DD.sub_temp.values
for ci, c in enumerate(["7A", "7B", "101A", "101B"]):
    v22 = DD[f"v22_a_{c}"].values; q18 = DD[f"v18_a_{c}"].values; base = v22.copy()
    for rec, Gr in DD.groupby("rec"):
        if rec in TNA8E and TNA8E[rec] not in hidE[int(Gr.fold.iloc[0])]:
            ix = Gr.index.values; q0 = q18[ix[Gr.hour.values == 0]][0]; base[ix] = q18[ix] + A7.BETA[Gr.hour.values] * np.clip(ST[idx[TNA8E[rec]], 23] - .36 - q0, -2, 2)
    anchE = set(DD.rec[np.abs(v22 - q18) > 1e-9]) | set(TNA8E)
    Qm = {r: g.sort_values("hour").set_index("hour")[f"v18_a_{c}"].reindex(range(24)).values for r, g in DD.groupby("rec")}
    a_, b_ = base.copy(), base.copy()
    for rec, Gr in DD.groupby("rec"):
        k = int(Gr.fold.iloc[0]); lab = lab_ev(rec)
        out = kf2(rec, pkE[k], Qm, lambda pv, k=k: bool(TRAIN[idx[pv]]) and pv not in hidE[k], {r for r in pkE[k] if r in Qm}, lab, lambda cur, lab=lab: hp2(idx[cur], lab))
        if not out: continue
        if ci == 0: fired_log.append(dict(layout="EV", rec=rec, steps=out[1], anchored=rec in anchE))
        ix = Gr.index.values; v = q18[ix] + SHAPE[Gr.hour.values] * out[0]
        if rec not in anchE: a_[ix] = v
        b_[ix] = v
    record("EV", "KF2a", ci, DD.day.values, y, base, a_, DD.farm.values); record("EV", "KF2b", ci, DD.day.values, y, base, b_, DD.farm.values)
print("done EV", flush=True)
FL = pd.DataFrame(fired_log); pd.set_option("display.width", 220)
print("fired by layout/steps/anchored:", FL[FL.layout.isin(["DIAG10", "EXT10", "EV"])].groupby(["layout", "steps", "anchored"]).size().to_dict())
Sx = pd.DataFrame(res); print(Sx.pivot_table(index=["arm", "layout"], columns="cell", values="pct").round(3).to_string())
print(Sx.groupby(["arm", "layout"]).p2.mean().round(3).unstack(0).to_string())
def bootP(L, a):
    v = blocks[(L, a)].values; ix = rng.integers(0, len(v), (20000, len(v))); m = v[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for a in ("KF2a", "KF2b"):
    ok = True; msg = []
    for L in sorted(set(Sx.layout)):
        c_ = Sx[(Sx.arm == a) & (Sx.layout == L)].pct; pw, pb = bootP(L, a)
        if L in ("DIAG10", "EXT10", "EV"):
            c = bool((c_ < 0).all()) and pw < .0125; ok &= c; msg.append(f"{L}[primary] cells {int((c_ < 0).sum())}/4 P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}")
        else:
            bad = c_.mean() > 0 and pb < .0125; ok &= not bad; msg.append(f"{L} mean {c_.mean():+.3f}% P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}")
    print(f"\n== KF2 {a}:"); print("\n".join("   " + m for m in msg)); print(f"KF2 {a}:", "ADOPT" if ok else "FAIL")
