# -*- coding: utf-8 -*-
"""SM1 (2026-10-14 집 클로드; 사용자: '작은 모델로 진행').  사전 고정(커밋 후 실행).  EC.  과거 정보만.  2차 기록만.
DG1(6.856, 학습 cert 쌍 + OOF 기본으로 잰 값 — 검증기 결과로 고르지 않음): 최적 당김 w — 1일 .92(보통 .95, 고EC .88), 2일 보통 .02·고EC .56, 3일 이상 ~0.
작은 상태 모델 = 시각별 지문 링커 앵커(LH3 P2 와 같은 링커: 날짜 d−1 후보, 0..h 지문, 확신 2등/1등 >= 1.5 & 거리 <= tau_h, cert 다음날 보이는 후보 제외) 에
  (1) 당김 w1 = .9 (1일 전 짝),  (2) 그 시각 1일 전 짝이 없으면 날짜 d−2 후보 중 같은 조건의 지문 짝을 찾아, 그 짝의 23시 EC >= 1.0(고EC 상태)일 때만 w2 = .5 로 당김.
  P_h = sg_h + w (A − cm_h).  기준 V7 = v7 확신 앵커(EA2).  보호 장치 없음(LH3 P2G 의 보호는 이득도 막았음).
판정(EC 분야 배정 규칙, 링커 의존, k = 1): (1) 주 P2LOO·EL1 3시드 모두 2차 dSSE < 0 AND (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록 없음);
  (2) EV 3시드 모두 전체 행 dSSE <= 0;  (3) DIAG10 손해 감시(시드 평균 악화 AND '후보 나음' 비율 < .025 이면 FAIL).
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261205)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); SUCC = {}; CERT = {}
for a_, b_, c_ in zip(TL.a1, TL.b, TL.cert):
    if c_: SUCC.setdefault(int(a_), []).append(int(b_)); CERT[int(b_)] = int(a_)
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
def dist(a, b, h): return float(np.nanmean(np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4)))
def tdate(b, lab):
    vi = np.where(lab)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; return d, pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
td_all = []
for b, a in CERT.items():
    lab = TRAIN.copy(); lab[b] = False; t = tdate(b, lab)
    if t is not None and DAYT[a] == t[0] - 1 and SEGT[a] == t[1]: td_all.append([dist(a, b, h) for h in range(24)])
TAU = np.nanquantile(np.array(td_all), .9, axis=0); print("tau_h (90% of true-pair distance):", np.round(TAU[[0, 6, 9, 12, 23]], 3), "n", len(td_all), flush=True)
def lh_picks(b, lab):
    """returns 24 x (pick index or None, w)"""
    t = tdate(b, lab)
    if t is None: return [(None, 0.0)] * 24
    def cands(gap):
        return [a for a in np.where(lab)[0] if a != b and DAYT[a] == t[0] - gap and SEGT[a] == t[1] and np.isfinite(EC[a, 23])
                and not any(lab[s] and DAYT[s] == t[0] - gap + 1 for s in SUCC.get(a, []))]
    c1, c2 = cands(1), cands(2); out = []
    def best(cand, h):
        if not cand: return None
        d = sorted((dist(a, b, h), a) for a in cand); ok = (len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)) and d[0][0] <= TAU[h]
        return d[0][1] if ok else None
    for h in range(24):
        a = best(c1, h)
        if a is not None: out.append((a, .9)); continue
        a2 = best(c2, h)
        out.append((a2, .5) if (a2 is not None and EC[a2, 23] >= 1.0) else (None, 0.0))
    return out
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
res, recd, adds, blocks = [], [], [], {}; PKC = {}
def run(Gv, y, sg, cm, recrows, v7val, hsf, tag, sd, ARM):
    base = sg.copy(); P = sg.copy(); hh = Gv.hour.values
    for r, ix in recrows.items():
        if r in v7val:
            base[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_); P[ix] = base[ix]; continue
        if NUM[idx[r]] < 179: continue
        hs = hsf(r); lab = TRAIN & ~np.array([rid(i) in hs for i in range(N)]) & np.isfinite(DM)
        pk = PKC.setdefault((tag, r, frozenset(hs)), lh_picks(idx[r], lab)); used = 0
        for j in ix:
            a, w = pk[hh[j]]
            if a is not None: P[j] = np.clip(sg[j] + w * (EC[a, 23] - cm[j]), lo_, hi_); used += 1
        if used: adds.append(dict(arm=ARM, val=tag, seed=str(sd), rec=r, share=used / len(ix), p2=NUM[idx[r]] >= 179, ok=(pk[23][0] == CERT.get(idx[r])) if (idx[r] in CERT and pk[23][0] is not None) else None))
    p2 = Gv.day.values >= 179; d = (P - y) ** 2 - (base - y) ** 2
    res.append(dict(arm=ARM, val=tag, seed=str(sd), dSSE_all=float(d.sum()), dSSE_p2=float(d[p2].sum()), all_pct=100 * (rm(P - y) / rm(base - y) - 1), p2_pct=100 * (rm((P - y)[p2]) / rm((base - y)[p2]) - 1)))
    bk = (Gv.farm + "_" + (Gv.day // 5).astype(str)).values
    blocks.setdefault((ARM, tag, "all"), []).append(pd.Series(d).groupby(bk).mean()); blocks.setdefault((ARM, tag, "p2"), []).append(pd.Series(d[p2]).groupby(bk[p2]).mean())
    for r_, v in pd.Series(d[p2]).groupby(Gv.rec.values[p2]).sum().items(): recd.append(dict(arm=ARM, val=tag, seed=str(sd), rec=r_, dsse=v))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict()
    hsf = lambda r, hidden=hidden, fold_of=fold_of: hidden.get(fold_of[r], set())
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hsf(r)}
    for ARM in ("SM1",): run(Gv, y, sg, cm, recrows, v7val, hsf, v, sd, ARM)
    print("done", v, sd, flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}
    v7val = {r: float(EC[idx[V7[r]], 23]) for r in set(Gv.rec[Gv.anchored.astype(bool)]) if r in V7}
    for ARM in ("SM1",): run(Gv, y, sg, cm, recrows, v7val, lambda r, fo=fo: hid[fo[r]], "EV", sd, ARM)
    print("done EV", sd, flush=True)
S = pd.DataFrame(res); A = pd.DataFrame(adds); RD = pd.DataFrame(recd); pd.set_option("display.width", 220)
print(S.round(3).to_string(index=False))
def boot(key):
    s_ = pd.concat(blocks[key], axis=1).mean(axis=1).values; ix = rng.integers(0, len(s_), (20000, len(s_))); m = s_[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for ARM in ("SM1",):
    SA = S[S.arm == ARM]; RA = RD[RD.arm == ARM]; AA = A[A.arm == ARM] if len(A) else A
    if len(AA): print(f"\n[{ARM}] added anchored pass-2 records per cell:", AA.groupby(["val", "seed"]).size().groupby("val").mean().round(1).to_dict(), "| mean hour share", round(float(AA.share.mean()), 2))
    main = SA[SA.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP2, _ = boot((ARM, "P2LOO", "p2")); harmed = RA[RA.val.isin(["P2LOO", "EL1"]) & (RA.dsse > 1e-6)]
    c1b = pP2 < .025 or len(harmed) == 0
    print(f"(1) main: cells better {int((main.dSSE_p2 < 0).sum())}/{len(main)} | P2LOO P(worse) {pP2:.4f} | harmed record-cells {len(harmed)} {harmed.groupby('rec').dsse.max().round(3).to_dict()} -> {'PASS' if c1 and c1b else 'FAIL'}")
    ev = SA[SA.val == "EV"]; c2 = bool((ev.dSSE_all <= 0).all()); print(f"(2) EV: all-rows dSSE {ev.dSSE_all.round(3).tolist()} -> {'PASS' if c2 else 'FAIL'}")
    dg = SA[SA.val == "DIAG10"]; _, pb = boot((ARM, "DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025); print(f"(3) DIAG10 guard: mean dSSE {dg.dSSE_all.mean():+.3f} share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
    g = RA.groupby(["val", "rec"]).dsse.mean(); print("   pass-2 record changes:", g[g.abs() > 1e-6].round(3).to_dict())
    print(f"SM1:", "ADOPT" if (c1 and c1b and c2 and c3) else "FAIL")
